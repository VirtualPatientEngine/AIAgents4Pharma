# Cell 1 Imports and Paths
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

MAX_TRACE_STRING_LENGTH = 2000
MAX_TRACE_LIST_ITEMS = 200
MAX_TRACE_DEPTH = 6

from statistics import mean, median, pstdev, stdev
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

BENCHMARK_JSON_PATH = Path("../benchmark_questions_set3.json")
# Set to a list of question IDs to run a focused sample; leave as None for the full set.
QUESTION_SAMPLE_IDS: Optional[List[str]] = None
BASE_DIR = Path(__file__).resolve().parent
TASK_COMPLETION_OUTPUT_PATH = Path("task_completion_set3_results.json")
TABLES_DIR = (BASE_DIR / "../tables").resolve()


# Cell 2 Load Benchmark Questions
def load_benchmark_questions(
    json_path: Path, selected_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    questions = (
        raw_payload.get("simulate_model_benchmark_questions")
        or raw_payload.get("get_modelinfo_benchmark_questions_set3")
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
thread_prefix = "task-completion-set3"

judge_model_name = "gpt-4o"

goldens: List[Golden] = [
    Golden(
        input=question["question"],
        name=question["id"],
        additional_metadata={
            "expected_answer": question.get("expected_answer"),
            "expected_tools": question.get("expected_tools"),
            "model_id": question.get("model_id"),
            "simulation_time": question.get("simulation_time"),
            "species": question.get("species"),
            "question_type": question.get("question_type"),
            "interval": question.get("interval"),
            "initial_concentration": question.get("initial_concentration"),
            "search_query": question.get("search_query"),
            "expected_count": question.get("expected_count"),
            "tool": question.get("tool"),
            "expected_values": question.get("expected_values"),
        },
        custom_column_key_values={
            "question_id": question["id"],
            "model_id": str(question.get("model_id"))
            if question.get("model_id") is not None
            else "",
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
            "interval": question_meta.get("interval"),
            "initial_concentration": question_meta.get("initial_concentration"),
            "search_query": question_meta.get("search_query"),
            "expected_count": question_meta.get("expected_count"),
            "tool": question_meta.get("tool"),
            "expected_values": question_meta.get("expected_values"),
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


def resolve_expected_path(expected_values: Optional[Dict[str, Any]], preferred_key: Optional[str] = None) -> Optional[Path]:
    if not expected_values:
        return None
    relative_path = None
    if preferred_key and preferred_key in expected_values:
        relative_path = expected_values.get(preferred_key)
    if relative_path is None and isinstance(expected_values, dict):
        # take the first value as a fallback
        for value in expected_values.values():
            relative_path = value
            break
    if not relative_path:
        return None

    candidate = (BASE_DIR / relative_path).resolve()
    if candidate.exists():
        return candidate

    alt_candidate = (TABLES_DIR / Path(relative_path).name).resolve()
    if alt_candidate.exists():
        return alt_candidate
    return None


def load_expected_dataframe(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def dataframe_from_payload(payload: Any) -> Optional[pd.DataFrame]:
    if payload is None:
        return None
    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in payload):
            return pd.DataFrame(payload)
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        try:
            df = pd.DataFrame(payload)
            if df.empty:
                df = pd.DataFrame([payload])
            return df
        except Exception:
            try:
                return pd.DataFrame([payload])
            except Exception:
                return None
    if isinstance(payload, str):
        candidate = resolve_expected_path({"payload": payload}, "payload")
        if candidate:
            return load_expected_dataframe(candidate)
    return None


def extract_dataframe_from_entries(
    entries: Any, candidate_keys: List[str]
) -> Optional[pd.DataFrame]:
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return dataframe_from_payload(entries)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # direct payload if the entry itself resembles a table
        direct_df = dataframe_from_payload(entry)
        if direct_df is not None and not direct_df.empty:
            return direct_df
        for key in candidate_keys:
            if key in entry:
                df = dataframe_from_payload(entry.get(key))
                if df is not None:
                    return df
    return None


def normalize_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    normalized = df.copy()
    normalized.columns = [str(col) for col in normalized.columns]
    normalized = normalized.reset_index(drop=True)
    try:
        normalized = normalized.sort_index(axis=1)
    except Exception:
        pass
    return normalized


def compare_numeric_content(
    actual_df: Optional[pd.DataFrame],
    expected_df: Optional[pd.DataFrame],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    if actual_df is None:
        return {"status": "missing_actual"}
    if expected_df is None:
        return {"status": "missing_expected"}

    act = normalize_dataframe(actual_df)
    exp = normalize_dataframe(expected_df)
    if act is None or exp is None:
        return {"status": "normalization_failed"}

    common_cols = [col for col in exp.columns if col in act.columns]
    numeric_common_cols: List[str] = []
    for col in common_cols:
        try:
            pd.to_numeric(exp[col], errors="raise")
            pd.to_numeric(act[col], errors="raise")
            numeric_common_cols.append(col)
        except Exception:
            continue

    if not numeric_common_cols:
        return {
            "status": "no_numeric_columns",
            "columns_considered": common_cols,
        }

    act_numeric = act[numeric_common_cols].apply(pd.to_numeric, errors="coerce")
    exp_numeric = exp[numeric_common_cols].apply(pd.to_numeric, errors="coerce")

    if act_numeric.shape != exp_numeric.shape:
        return {
            "status": "shape_mismatch",
            "actual_shape": act_numeric.shape,
            "expected_shape": exp_numeric.shape,
        }

    diff = (act_numeric - exp_numeric).abs()
    max_diff = diff.to_numpy()[~pd.isna(diff.to_numpy())].max() if diff.size else 0.0
    match = bool((diff.fillna(0) <= tolerance).all().all())

    return {
        "status": "match" if match else "mismatch",
        "max_abs_diff": float(max_diff) if pd.notna(max_diff) else None,
        "columns_compared": numeric_common_cols,
        "actual_preview": act_numeric.head().to_dict(orient="list"),
        "expected_preview": exp_numeric.head().to_dict(orient="list"),
    }


def parse_species_from_question(question_text: str) -> Optional[str]:
    if not question_text:
        return None
    marker = "species "
    if marker in question_text.lower():
        lower_text = question_text.lower()
        start = lower_text.find(marker)
        if start != -1:
            start += len(marker)
            remainder = question_text[start:]
            tokens = remainder.split(" in ")
            candidate = tokens[0].strip(" ?.")
            return candidate
    return None


def lookup_species_value(df: Optional[pd.DataFrame], species_name: Optional[str]) -> Optional[float]:
    if df is None or species_name is None:
        return None
    columns = {str(col).lower(): col for col in df.columns}
    name_col = None
    for key in ("display_name", "species", "name"):
        if key in columns:
            name_col = columns[key]
            break
    value_col = None
    for key in ("steady_state_concentration", "concentration", "value"):
        matches = [col for col in columns if key in col]
        if matches:
            value_col = columns[matches[0]]
            break
    if name_col is None or value_col is None:
        return None
    try:
        mask = (
            df[name_col]
            .astype(str)
            .str.strip()
            .str.lower()
            == species_name.strip().lower()
        )
        if mask.any():
            return float(df.loc[mask, value_col].iloc[-1])
    except Exception:
        return None
    return None


def evaluate_steady_state_artifact(
    question: Dict[str, Any], raw_state: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    expected_path = resolve_expected_path(
        question.get("expected_values"), "steady_state"
    )
    expected_df = load_expected_dataframe(expected_path)
    actual_entries = []
    if isinstance(raw_state, dict):
        actual_entries = raw_state.get("dic_steady_state_data", [])
    actual_df = extract_dataframe_from_entries(
        actual_entries,
        ["steady_state_table", "table", "data", "steady_state"],
    )
    comparison = compare_numeric_content(actual_df, expected_df)

    species_name = question.get("species") or parse_species_from_question(
        question.get("question", "")
    )
    expected_species_value = lookup_species_value(expected_df, species_name)
    actual_species_value = lookup_species_value(actual_df, species_name)

    species_evaluation = None
    if species_name:
        species_evaluation = {
            "species": species_name,
            "expected_value": expected_species_value,
            "actual_value": actual_species_value,
            "abs_diff": None
            if actual_species_value is None or expected_species_value is None
            else abs(actual_species_value - expected_species_value),
        }

    return {
        "question_type": "steady_state",
        "status": comparison.get("status"),
        "expected_artifact": str(expected_path) if expected_path else None,
        "comparison": comparison,
        "species_evaluation": species_evaluation,
    }


def evaluate_parameter_scan_artifact(
    question: Dict[str, Any], raw_state: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    expected_path = resolve_expected_path(
        question.get("expected_values"), "parameter_scan"
    )
    expected_df = load_expected_dataframe(expected_path)
    actual_entries = []
    if isinstance(raw_state, dict):
        actual_entries = raw_state.get("dic_scanned_data", [])
        if not actual_entries:
            actual_entries = raw_state.get("dic_simulated_data", [])
    actual_df = extract_dataframe_from_entries(
        actual_entries,
        ["scan_table", "scan_result", "data", "table"],
    )
    comparison = compare_numeric_content(actual_df, expected_df, tolerance=1e-5)

    return {
        "question_type": "parameter_scan",
        "status": comparison.get("status"),
        "expected_artifact": str(expected_path) if expected_path else None,
        "comparison": comparison,
    }


def evaluate_artifact(question: Dict[str, Any], raw_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if raw_state is None:
        return {"status": "raw_state_unavailable"}
    question_type = question.get("question_type")
    if question_type == "steady_state":
        return evaluate_steady_state_artifact(question, raw_state)
    if question_type == "parameter_scan":
        return evaluate_parameter_scan_artifact(question, raw_state)
    return {"status": "not_applicable", "question_type": question_type}


def reset_traces() -> None:
    trace_manager.clear_traces()


t2b_runner = T2BAgentRunner(base_thread_prefix=thread_prefix, llm=llm_model)

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
    artifact_eval = evaluate_artifact(question, raw_state)
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
            "question_type": question.get("question_type"),
            "interval": question.get("interval"),
            "initial_concentration": question.get("initial_concentration"),
            "species": question.get("species"),
            "search_query": question.get("search_query"),
            "expected_count": question.get("expected_count"),
            "tool": question.get("tool"),
            "expected_values": question.get("expected_values"),
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
            "question_type": question.get("question_type"),
            "artifact_evaluation": artifact_eval,
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
            "interval": question_lookup.get(result.question_id, {}).get("interval"),
            "initial_concentration": question_lookup.get(result.question_id, {}).get(
                "initial_concentration"
            ),
            "expected_values": question_lookup.get(result.question_id, {}).get(
                "expected_values"
            ),
            "search_query": question_lookup.get(result.question_id, {}).get(
                "search_query"
            ),
            "expected_count": question_lookup.get(result.question_id, {}).get(
                "expected_count"
            ),
            "tool": question_lookup.get(result.question_id, {}).get("tool"),
            "artifact_evaluation": metric_details[idx].get("artifact_evaluation"),
        }
        for idx, result in enumerate(run_results)
    ],
}

# derive overall artifact stats
artifact_stats = {
    "total": len([res for res in artifact_results if res]),
    "evaluated": 0,
    "matches": 0,
    "mismatches": 0,
    "missing_actual": 0,
    "missing_expected": 0,
    "shape_mismatch": 0,
    "normalization_failed": 0,
    "no_numeric_columns": 0,
    "not_applicable": 0,
    "raw_state_unavailable": 0,
}

for res in artifact_results:
    if not isinstance(res, dict):
        continue
    status = res.get("status")
    if status == "not_applicable":
        artifact_stats["not_applicable"] += 1
        continue
    if status == "raw_state_unavailable":
        artifact_stats["raw_state_unavailable"] += 1
        continue

    artifact_stats["evaluated"] += 1

    if status == "missing_actual":
        artifact_stats["missing_actual"] += 1
        continue
    if status == "missing_expected":
        artifact_stats["missing_expected"] += 1
        continue
    if status in {"normalization_failed", "shape_mismatch", "no_numeric_columns"}:
        artifact_stats[status] += 1
        continue

    comparison = res.get("comparison", {})
    comp_status = comparison.get("status")
    if comp_status == "match":
        artifact_stats["matches"] += 1
    elif comp_status:
        artifact_stats["mismatches"] += 1

task_completion_summary["artifact_summary"] = artifact_stats

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
