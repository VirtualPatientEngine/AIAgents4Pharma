# Cell 1 Imports and Paths
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from statistics import mean, median, pstdev, stdev
from statistics import StatisticsError

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe
from deepeval.tracing.tracing import trace_manager
from deepeval.test_case import LLMTestCase
from deepeval.utils import dataclass_to_dict
from langchain_openai import ChatOpenAI

try:  # LangChain >= 0.1.0
    from langchain_core.messages import BaseMessage
except ImportError:  # Backwards compatibility
    from langchain.schema import BaseMessage  # type: ignore

from aiagents4pharma.talk2biomodels.agents.t2b_agent import (
    get_app as get_t2b_agent,
)
from aiagents4pharma.talk2biomodels.states.state_talk2biomodels import Talk2Biomodels

BENCHMARK_JSON_PATH = Path("../benchmark_questions_set1.json")
# Set to a list of question IDs to run a focused sample; leave as None for the full set.
QUESTION_SAMPLE_IDS: Optional[List[str]] = None


# Cell 2 Load Benchmark Questions
def load_benchmark_questions(
    json_path: Path, selected_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    questions = raw_payload["simulate_model_benchmark_questions"]
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
thread_prefix = "task-completion-set1"
t2b_agent = get_t2b_agent(thread_prefix, llm_model)

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
        },
        custom_column_key_values={
            "question_id": question["id"],
            "model_id": question.get("model_id"),
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
    def __init__(self, agent, base_thread_prefix: str):
        self.agent = agent
        self.base_thread_prefix = base_thread_prefix

    @observe(type="agent")
    def invoke(self, *, question_text: str, question_id: str) -> Dict[str, Any]:
        state = build_initial_state(question_text)
        invocation_thread = f"{self.base_thread_prefix}-{question_id}"
        result_state = self.agent.invoke(
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

        serializable_state = {
            "model_id": result_state.get("model_id", []),
            "sbml_file_path": result_state.get("sbml_file_path", []),
            "dic_simulated_data": result_state.get("dic_simulated_data", []),
            "dic_scanned_data": result_state.get("dic_scanned_data", []),
            "dic_steady_state_data": result_state.get("dic_steady_state_data", []),
            "dic_annotations_data": result_state.get("dic_annotations_data", []),
        }

        return {
            "answer": answer,
            "assistant_messages": assistant_messages,
            "all_messages": normalized_messages,
            "state_fields": serializable_state,
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


def reset_traces() -> None:
    trace_manager.clear_traces()


t2b_runner = T2BAgentRunner(t2b_agent, thread_prefix)

# Cell 5 Execute Evaluation Loop (Sample Run)
run_results: List[AgentRunResult] = []
metric_scores: List[float] = []
metric_reasons: List[str] = []
metric_details: List[Dict[str, Any]] = []

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

    trace_obj = get_latest_trace()
    trace_tree = build_trace_dict(trace_obj)
    serialized_trace = dataclass_to_dict(trace_obj) if trace_obj else None

    metric_instance = TaskCompletionMetric(
        model=judge_model_name,
        include_reason=True,
        async_mode=False,
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
    if trace_tree is not None:
        test_case._trace_dict = trace_tree

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
            trace_tree=trace_tree,
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
        }
        for idx, result in enumerate(run_results)
    ],
}

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
