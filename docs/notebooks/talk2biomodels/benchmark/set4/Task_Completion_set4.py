# Cell 1 Imports and Paths
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_TRACE_STRING_LENGTH = 2000
MAX_TRACE_LIST_ITEMS = 200
MAX_TRACE_DEPTH = 6

from statistics import median, pstdev, stdev
from statistics import StatisticsError

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe
from deepeval.tracing.tracing import trace_manager
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI

try:  # LangChain >= 0.1.0
    from langchain_core.messages import BaseMessage
except ImportError:  # Backwards compatibility
    from langchain.schema import BaseMessage  # type: ignore

from aiagents4pharma.talk2biomodels.agents.t2b_agent import (
    get_app as get_t2b_agent,
)
from aiagents4pharma.talk2biomodels.states.state_talk2biomodels import Talk2Biomodels

BENCHMARK_JSON_PATH = Path("../benchmark_questions_set4.json")
# Set to a list of question IDs to run a focused sample; leave as None for the full set.
QUESTION_SAMPLE_IDS: Optional[List[str]] = None
TASK_COMPLETION_OUTPUT_PATH = Path("task_completion_set4_results.json")


# Cell 2 Load Benchmark Questions
def load_benchmark_questions(
    json_path: Path, selected_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    questions = (
        raw_payload.get("get_annotation_benchmark_questions_set4")
        or raw_payload.get("simulate_model_benchmark_questions")
        or []
    )
    if selected_ids is None:
        return questions

    question_by_id = {question["id"]: question for question in questions}
    ordered_subset = [
        question_by_id[qid] for qid in selected_ids if qid in question_by_id
    ]
    return ordered_subset


benchmark_questions = load_benchmark_questions(BENCHMARK_JSON_PATH, QUESTION_SAMPLE_IDS)
question_lookup: Dict[str, Dict[str, Any]] = {
    item["id"]: item for item in benchmark_questions
}


def _safe_stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return stdev(values)
    except StatisticsError:
        return 0.0


# Cell 3 Initialize Models and Metric
llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
thread_prefix = "task-completion-set4"

judge_model_name = "gpt-4o"

goldens: List[Golden] = [
    Golden(
        input=question["question"],
        name=question["id"],
        additional_metadata={
            "expected_answer": question.get("expected_answer"),
            "expected_tools": question.get("expected_tools"),
            "model_id": question.get("model_id"),
            "question_type": question.get("question_type"),
            "expected_values": question.get("expected_values"),
        },
        custom_column_key_values={
            "question_id": question["id"],
            "model_id": (
                str(question.get("model_id"))
                if question.get("model_id") is not None
                else ""
            ),
            "question_type": question.get("question_type"),
        },
    )
    for question in benchmark_questions
]

dataset = EvaluationDataset(goldens=goldens)


# Cell 4 Helper Functions and Runner
@dataclass
class AgentRunResult:
    question_id: str
    answer: Optional[str]
    assistant_messages: List[Dict[str, Any]]
    all_messages: List[Dict[str, Any]]
    state_fields: Dict[str, Any]
    trace: Optional[Dict[str, Any]]
    trace_tree: Optional[Dict[str, Any]]
    thread_id: str


def build_initial_state(question_text: str) -> Talk2Biomodels:
    state = Talk2Biomodels(
        llm_model=llm_model,
        text_embedding_model=None,
        pdf_file_name="",
        model_id=[],
        sbml_file_path=[],
        dic_simulated_data=[],
        dic_scanned_data=[],
        dic_steady_state_data=[],
        dic_annotations_data=[],
    )
    state["messages"] = [{"role": "user", "content": question_text}]
    return state


class T2BAgentRunner:
    def __init__(self, *, base_thread_prefix: str, llm):
        self.base_thread_prefix = base_thread_prefix
        self.llm = llm

    @observe(type="agent")
    def invoke(self, *, question_text: str, question_id: str) -> Dict[str, Any]:
        state = build_initial_state(question_text)
        invocation_thread = f"{self.base_thread_prefix}-{question_id}"
        agent = get_t2b_agent(invocation_thread, self.llm)
        result_state = agent.invoke(
            state,
            config={"configurable": {"thread_id": invocation_thread}},
        )

        messages = result_state.get("messages", [])
        normalized_messages: List[Dict[str, Any]] = [
            self._normalize_message(msg) for msg in messages
        ]
        assistant_messages = [
            msg for msg in normalized_messages if msg.get("role") in {"assistant", "ai"}
        ]
        answer = assistant_messages[-1].get("content") if assistant_messages else None

        question_meta = question_lookup.get(question_id, {})
        serializable_state = {
            "model_id": result_state.get("model_id", []),
            "sbml_file_path": result_state.get("sbml_file_path", []),
            "dic_simulated_data": summarize_value(
                result_state.get("dic_simulated_data", [])
            ),
            "dic_scanned_data": summarize_value(
                result_state.get("dic_scanned_data", [])
            ),
            "dic_steady_state_data": summarize_value(
                result_state.get("dic_steady_state_data", [])
            ),
            "dic_annotations_data": summarize_value(
                result_state.get("dic_annotations_data", [])
            ),
            "question_type": question_meta.get("question_type"),
            "expected_values": question_meta.get("expected_values"),
            "model_id_expected": question_meta.get("model_id"),
        }

        return {
            "answer": answer,
            "assistant_messages": assistant_messages,
            "all_messages": normalized_messages,
            "state_fields": serializable_state,
            "thread_id": invocation_thread,
            "raw_state": result_state,
        }

    @staticmethod
    def _normalize_message(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict):
            return {
                "role": message.get("role"),
                "content": message.get("content"),
                **{k: v for k, v in message.items() if k not in {"role", "content"}},
            }

        if isinstance(message, BaseMessage):
            payload = {
                "role": getattr(message, "type", None),
                "content": getattr(message, "content", None),
            }
            additional = getattr(message, "additional_kwargs", None)
            if isinstance(additional, dict):
                payload.update(additional)
            return payload

        return {"role": type(message).__name__, "content": str(message)}


def get_latest_trace() -> Optional[Any]:
    traces = trace_manager.get_all_traces_dict()
    if traces:
        return traces[-1]
    return None


def build_trace_dict(trace_obj: Any) -> Optional[Dict[str, Any]]:
    if trace_obj is None:
        return None
    root_spans = getattr(trace_obj, "root_spans", None)
    if not root_spans:
        return None
    root_span = root_spans[0]
    return trace_manager.create_nested_spans_dict(root_span)


def summarize_value(value: Any) -> Any:
    if isinstance(value, list):
        summary: Dict[str, Any] = {"type": "list", "length": len(value)}
        if value:
            summary["preview"] = summarize_value(value[0])
        return summary

    if isinstance(value, dict):
        keys = list(value.keys())
        return {
            "type": "dict",
            "key_count": len(keys),
            "keys_preview": keys[:10],
        }

    return value


def _truncate_trace_value(value: Any, depth: int = 0) -> Any:
    if depth >= MAX_TRACE_DEPTH:
        return "<truncated depth>"

    if isinstance(value, str):
        if len(value) > MAX_TRACE_STRING_LENGTH:
            return value[:MAX_TRACE_STRING_LENGTH] + "... <truncated>"
        return value

    if isinstance(value, list):
        if not value:
            return []
        truncated = [
            _truncate_trace_value(item, depth + 1)
            for item in value[:MAX_TRACE_LIST_ITEMS]
        ]
        if len(value) > MAX_TRACE_LIST_ITEMS:
            truncated.append(
                f"<truncated {len(value) - MAX_TRACE_LIST_ITEMS} additional items>"
            )
        return truncated

    if isinstance(value, dict):
        pruned: Dict[str, Any] = {}
        for key, item in value.items():
            if key in {"output", "input"}:
                pruned[key] = summarize_value(item)
            else:
                pruned[key] = _truncate_trace_value(item, depth + 1)
        return pruned

    return value


def sanitize_trace_tree(
    trace_tree: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if trace_tree is None:
        return None
    return _truncate_trace_value(trace_tree)


def reset_traces() -> None:
    trace_manager.clear_traces()


t2b_runner = T2BAgentRunner(base_thread_prefix=thread_prefix, llm=llm_model)


def _to_stripped(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip()


def _normalize_annotation_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if (cleaned.startswith("[") and cleaned.endswith("]")) or (
            cleaned.startswith("{") and cleaned.endswith("}")
        ):
            try:
                parsed = json.loads(cleaned)
                return _normalize_annotation_values(parsed)
            except json.JSONDecodeError:
                pass
        if ":" in cleaned:
            cleaned = cleaned.split(":", 1)[1].strip() or cleaned
        segments = [seg.strip() for seg in cleaned.replace(";", ",").split(",")]
        return sorted({seg for seg in segments if seg})
    if isinstance(value, (list, tuple, set)):
        collected: List[str] = []
        for item in value:
            collected.extend(_normalize_annotation_values(item))
        return sorted({item for item in collected if item})
    if isinstance(value, dict):
        collected: List[str] = []
        for item in value.values():
            collected.extend(_normalize_annotation_values(item))
        return sorted({item for item in collected if item})
    return _normalize_annotation_values(str(value))


def _match_string_case_insensitive(candidate: Optional[str], target: str) -> bool:
    if candidate is None:
        return False
    return candidate.strip().lower() == target.strip().lower()


def _extract_species_annotations(payload: Any, species_name: str) -> List[str]:
    if payload is None:
        return []

    collected: List[str] = []
    stack: List[Any] = [payload]
    normalized_target = species_name.strip().lower()

    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            # direct match on key
            for key, value in current.items():
                if isinstance(key, str) and key.strip().lower() == normalized_target:
                    collected.extend(_normalize_annotation_values(value))
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)
            # heuristic match based on dedicated fields
            name_fields = [
                "species",
                "species_name",
                "name",
                "display_name",
                "entity",
            ]
            matched = False
            for field in name_fields:
                candidate = current.get(field)
                if isinstance(candidate, (list, tuple)):
                    if any(
                        _match_string_case_insensitive(item, normalized_target)
                        for item in candidate
                        if isinstance(item, str)
                    ):
                        matched = True
                        break
                elif isinstance(candidate, str) and _match_string_case_insensitive(
                    candidate, normalized_target
                ):
                    matched = True
                    break
            if matched:
                for key, value in current.items():
                    if key in name_fields:
                        continue
                    if isinstance(value, (dict, list, tuple)):
                        collected.extend(_normalize_annotation_values(value))
                    elif isinstance(value, str):
                        collected.extend(_normalize_annotation_values(value))
        elif isinstance(current, (list, tuple, set)):
            stack.extend(list(current))
        elif isinstance(current, str):
            text = current.strip()
            if not text:
                continue
            lowered = text.lower()
            if normalized_target in lowered:
                if ":" in text:
                    prefix, suffix = text.split(":", 1)
                    if _match_string_case_insensitive(prefix, normalized_target):
                        collected.extend(_normalize_annotation_values(suffix))
                else:
                    tokens = [token.strip() for token in text.split() if token.strip()]
                    if tokens and _match_string_case_insensitive(
                        tokens[0], normalized_target
                    ):
                        collected.extend(
                            _normalize_annotation_values(" ".join(tokens[1:]))
                        )

    return sorted({item for item in collected if item})


def _normalize_expected_map(expected: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    if not isinstance(expected, dict):
        return normalized
    for species, value in expected.items():
        if species is None:
            continue
        normalized_species = str(species).strip()
        if not normalized_species:
            continue
        normalized[normalized_species] = _normalize_annotation_values(value)
    return normalized


def evaluate_annotations_artifact(
    question: Dict[str, Any], raw_state: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    expected_map_raw = question.get("expected_values")
    expected_map = _normalize_expected_map(expected_map_raw)
    if not expected_map:
        return {"status": "not_applicable"}
    if raw_state is None:
        return {"status": "raw_state_unavailable"}

    annotations_entries: Any = raw_state.get("dic_annotations_data")
    if isinstance(annotations_entries, dict):
        annotations_entries = [annotations_entries]
    if not annotations_entries:
        return {"status": "missing_actual"}

    if isinstance(annotations_entries, list) and annotations_entries:
        latest_payload = annotations_entries[-1]
    else:
        latest_payload = annotations_entries

    per_species: List[Dict[str, Any]] = []
    overall_match = True
    actual_map: Dict[str, List[str]] = {}

    for species, expected_values in expected_map.items():
        actual_values = _extract_species_annotations(latest_payload, species)
        actual_map[species] = actual_values
        species_match = bool(actual_values) and (
            set(actual_values) == set(expected_values)
            if expected_values
            else not actual_values
        )
        overall_match = overall_match and species_match
        per_species.append(
            {
                "species": species,
                "expected": expected_values,
                "actual": actual_values,
                "status": (
                    "match"
                    if species_match
                    else ("missing" if not actual_values else "mismatch")
                ),
            }
        )

    return {
        "status": "match" if overall_match else "mismatch",
        "expected": expected_map,
        "actual": actual_map,
        "per_species": per_species,
    }


# Cell 5 Execute Evaluation Loop (Sample Run)
run_results: List[AgentRunResult] = []
metric_scores: List[float] = []
metric_reasons: List[str] = []
metric_details: List[Dict[str, Any]] = []
artifact_results: List[Dict[str, Any]] = []

total_questions = len(benchmark_questions)

for question in benchmark_questions:
    question_id = question["id"]
    question_text = question["question"]

    current_index = len(run_results) + 1
    print(f"Processing {current_index:03d}/{total_questions:03d}: {question_id}")

    reset_traces()

    agent_payload = t2b_runner.invoke(
        question_text=question_text,
        question_id=question_id,
    )
    thread_id = agent_payload["thread_id"]

    print(f"→ Thread {current_index:03d}: {thread_id} (question={question_id})")

    raw_state = agent_payload.pop("raw_state", None)
    artifact_eval = evaluate_annotations_artifact(question, raw_state)
    artifact_results.append(artifact_eval)

    trace_obj = get_latest_trace()
    trace_tree = build_trace_dict(trace_obj)
    sanitized_trace_tree = sanitize_trace_tree(trace_tree)
    serialized_trace = sanitized_trace_tree

    metric_instance = TaskCompletionMetric(
        model=judge_model_name,
        include_reason=True,
        async_mode=False,
        threshold=0.5,
        verbose_mode=True,
    )

    test_case = LLMTestCase(
        input=question_text,
        actual_output=agent_payload["answer"],
        expected_output=question.get("expected_answer"),
        additional_metadata={
            "model_id": question.get("model_id"),
            "expected_tools": question.get("expected_tools"),
            "simulation_time": question.get("simulation_time"),
            "species": question.get("species"),
        },
        name=question_id,
    )
    if sanitized_trace_tree is not None:
        test_case._trace_dict = sanitized_trace_tree

    score = metric_instance.measure(test_case)
    reason = metric_instance.reason or ""

    run_results.append(
        AgentRunResult(
            question_id=question_id,
            answer=agent_payload["answer"],
            assistant_messages=agent_payload["assistant_messages"],
            all_messages=agent_payload["all_messages"],
            state_fields=agent_payload["state_fields"],
            trace=serialized_trace,
            trace_tree=sanitized_trace_tree,
            thread_id=thread_id,
        )
    )

    metric_scores.append(score)
    metric_reasons.append(reason)
    metric_details.append(
        {
            "question_id": question_id,
            "task": getattr(metric_instance, "task", None),
            "outcome": getattr(metric_instance, "outcome", None),
            "verdict": getattr(metric_instance, "verdict", None),
            "reason": reason,
            "verbose_logs": getattr(metric_instance, "verbose_logs", None),
            "artifact_evaluation": artifact_eval,
            "question_type": question.get("question_type"),
        }
    )

    print(f"[Task Completion] {question_id}: score={score:.3f}")

    reset_traces()

# Cell 6 Aggregate Results
average_task_completion = (
    sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
)

task_completion_summary = {
    "question_count": len(run_results),
    "average_score": average_task_completion,
    "median_score": median(metric_scores) if metric_scores else 0.0,
    "stdev_sample": _safe_stdev(metric_scores),
    "stdev_population": pstdev(metric_scores) if len(metric_scores) > 1 else 0.0,
    "minimum_score": min(metric_scores) if metric_scores else 0.0,
    "maximum_score": max(metric_scores) if metric_scores else 0.0,
    "pass_rate_threshold_0.5": (
        sum(score >= 0.5 for score in metric_scores) / len(metric_scores)
        if metric_scores
        else 0.0
    ),
    "pass_rate_threshold_0.7": (
        sum(score >= 0.7 for score in metric_scores) / len(metric_scores)
        if metric_scores
        else 0.0
    ),
    "per_question": [
        {
            "question_id": result.question_id,
            "score": metric_scores[idx],
            "reason": metric_reasons[idx],
            "verdict": metric_details[idx]["verdict"],
            "task": metric_details[idx]["task"],
            "outcome": metric_details[idx]["outcome"],
            "verbose_logs": metric_details[idx]["verbose_logs"],
            "answer": result.answer,
            "trace_available": result.trace is not None,
            "thread_id": result.thread_id,
            "question_type": question_lookup.get(result.question_id, {}).get(
                "question_type"
            ),
            "model_id": question_lookup.get(result.question_id, {}).get("model_id"),
            "expected_values": question_lookup.get(result.question_id, {}).get(
                "expected_values"
            ),
            "artifact_evaluation": metric_details[idx]["artifact_evaluation"],
        }
        for idx, result in enumerate(run_results)
    ],
}

artifact_summary = {
    "total": len(artifact_results),
    "match": 0,
    "mismatch": 0,
    "missing_actual": 0,
    "raw_state_unavailable": 0,
    "not_applicable": 0,
}

for evaluation in artifact_results:
    if not isinstance(evaluation, dict):
        continue
    status = evaluation.get("status")
    if status in artifact_summary:
        artifact_summary[status] += 1
    else:
        artifact_summary.setdefault(status, 0)
        artifact_summary[status] += 1

task_completion_summary["artifact_summary"] = artifact_summary

print("Task Completion evaluation complete.")
print(
    f"Evaluated {task_completion_summary['question_count']} questions. "
    f"Average score: {task_completion_summary['average_score']:.3f}"
)
print(f"Median score: {task_completion_summary['median_score']:.3f}")
print(
    f"Sample stdev: {task_completion_summary['stdev_sample']:.3f} | "
    f"Population stdev: {task_completion_summary['stdev_population']:.3f}"
)
print(
    f"Min score: {task_completion_summary['minimum_score']:.3f} | "
    f"Max score: {task_completion_summary['maximum_score']:.3f}"
)
print(
    f"Pass rate >=0.5: {task_completion_summary['pass_rate_threshold_0.5']:.3f} | "
    f"Pass rate >=0.7: {task_completion_summary['pass_rate_threshold_0.7']:.3f}"
)

output_payload = {
    "summary": task_completion_summary,
    "scores": metric_scores,
    "reasons": metric_reasons,
    "artifacts": artifact_results,
}

with TASK_COMPLETION_OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)

print(f"Task Completion results saved to {TASK_COMPLETION_OUTPUT_PATH}")
