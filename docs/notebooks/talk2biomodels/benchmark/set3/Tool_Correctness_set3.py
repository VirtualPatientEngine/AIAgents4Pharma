# Cell 1 Imports and Paths
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from statistics import median, pstdev, stdev
from statistics import StatisticsError

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.metrics.tool_correctness.tool_correctness import ToolCallParams
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe
from deepeval.tracing.tracing import trace_manager
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

BENCHMARK_JSON_PATH = Path("../benchmark_questions_set3.json")
QUESTION_SAMPLE_IDS: Optional[List[str]] = None
TOOL_CORRECTNESS_OUTPUT_PATH = Path("tool_correctness_set3_results.json")


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


# Cell 3 Initialize LLM
llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
thread_prefix = "tool-correctness-set3"


# Cell 4 Helper Functions
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


def normalize_message(message: Any) -> Dict[str, Any]:
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


def reset_traces() -> None:
    trace_manager.clear_traces()


def get_latest_trace() -> Optional[Any]:
    traces = trace_manager.get_all_traces()
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


def _strip_none_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_species_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if item is not None]
        return sorted(set(cleaned)) if cleaned else None
    return [str(value).strip()]


def _normalize_numeric_sequence(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return [
            _to_float(v)
            for _, v in sorted(value.items(), key=lambda item: str(item[0]))
        ]
    if not isinstance(value, (list, tuple)):
        return [_to_float(value)]
    normalized: List[float] = []
    for item in value:
        normalized.append(_to_float(item))
    return normalized


def parse_species_from_question(question_text: str) -> Optional[str]:
    if not question_text:
        return None
    lower = question_text.lower()
    marker = "species "
    if marker in lower:
        start = lower.find(marker) + len(marker)
        remainder = question_text[start:]
        token = remainder.split(" in ")[0]
        return token.strip(" ?.")
    return None


def _canon_simulate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    sys_bio_model = params.get("sys_bio_model", {})
    arg_data = params.get("arg_data", {})
    time_data = arg_data.get("time_data", {})

    raw_duration = time_data.get("duration")
    duration = _to_float(raw_duration)

    interval = _to_float(time_data.get("interval"))

    species_data = arg_data.get("species_to_be_analyzed_before_experiment")
    if not species_data:
        species_data = params.get("species_to_be_analyzed_before_experiment")

    species_name = None
    species_concentration = None
    if isinstance(species_data, dict):
        names = species_data.get("species_name")
        concs = species_data.get("species_concentration")

        species_names = _normalize_species_list(names)
        if species_names:
            species_name = species_names[0] if len(species_names) == 1 else species_names

        if isinstance(concs, (list, tuple)):
            species_concentration = _to_float(concs[0]) if concs else None
        else:
            species_concentration = _to_float(concs)

    return _strip_none_values(
        {
            "model_id": (
                str(sys_bio_model.get("biomodel_id"))
                if sys_bio_model.get("biomodel_id") is not None
                else None
            ),
            "duration": duration,
            "interval": interval,
            "species": species_name,
            "initial_concentration": species_concentration,
        }
    )


def _canon_steady_state_params(
    params: Dict[str, Any], question: Optional[Dict[str, Any]], for_expected: bool
) -> Dict[str, Any]:
    species = params.get("species_names") or params.get("species")
    species_list = _normalize_species_list(species)
    if species_list is None and for_expected and question is not None:
        fallback = question.get("species") or parse_species_from_question(
            question.get("question", "")
        )
        species_list = _normalize_species_list(fallback)

    model_id = params.get("model_id")
    if model_id is None and for_expected and question is not None:
        model_id = question.get("model_id")

    return _strip_none_values(
        {
            "model_id": str(model_id) if model_id is not None else None,
            "species": species_list,
        }
    )


def _canon_parameter_scan_params(
    params: Dict[str, Any], question: Optional[Dict[str, Any]], for_expected: bool
) -> Dict[str, Any]:
    source = {}
    if isinstance(params, dict):
        source.update(params)
    if for_expected and question is not None:
        arguments = question.get("arguments", {})
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                source.setdefault(key, value)

    canonical = {
        "species_parameter_name": source.get("species_parameter_name"),
        "species_parameter_values": _normalize_numeric_sequence(
            source.get("species_parameter_values")
        ),
        "stepsize_scan": _to_float(source.get("stepsize_scan")),
        "time": _to_float(source.get("time")),
        "intervals": _to_float(source.get("intervals")),
        "filename": None,
        "species_names": _normalize_species_list(source.get("species_names")),
    }

    filename = source.get("filename") or source.get("output_filename")
    if filename:
        canonical["filename"] = Path(str(filename)).name

    return _strip_none_values(canonical)


def canonicalize_tool_input(
    tool_name: Optional[str],
    params: Any,
    question: Optional[Dict[str, Any]],
    *,
    for_expected: bool = False,
) -> Dict[str, Any]:
    if params is None:
        params_dict: Dict[str, Any] = {}
    elif isinstance(params, dict):
        params_dict = params.copy()
    else:
        return {}

    if tool_name == "simulate_model":
        return _canon_simulate_model_params(params_dict)

    if tool_name == "steady_state":
        return _canon_steady_state_params(params_dict, question, for_expected)

    if tool_name == "parameter_scan":
        return _canon_parameter_scan_params(params_dict, question, for_expected)

    if tool_name == "search_models":
        canonical = {
            "query": params_dict.get("query") or params_dict.get("search_query")
        }
        number = params_dict.get("number") or params_dict.get("top_k")
        if number is not None:
            canonical["number"] = _to_float(number)
        if for_expected and question is not None:
            canonical.setdefault("query", question.get("search_query"))
            canonical.setdefault("number", _to_float(question.get("expected_count")))
        return _strip_none_values(canonical)

    if tool_name == "ask_question":
        return {}

    return _strip_none_values(params_dict)


def _parse_simulation_duration(simulation_time: Optional[str]) -> Optional[float]:
    if not simulation_time:
        return None
    matches = re.findall(r"[0-9]+(?:\.[0-9]+)?", simulation_time)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _parse_input_parameters(raw_input: Any) -> Any:
    if raw_input is None:
        return None

    if isinstance(raw_input, str):
        try:
            return json.loads(raw_input)
        except Exception:
            return raw_input

    return raw_input


def extract_tool_events_from_trace(
    trace_tree: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not trace_tree:
        return []

    extracted: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = [trace_tree]

    while stack:
        node = stack.pop()
        children = node.get("children", []) or []
        stack.extend(children)

        tools_called = node.get("toolsCalled") or node.get("tools_called") or []
        if tools_called:
            for tool_entry in tools_called:
                if isinstance(tool_entry, dict):
                    extracted.append(tool_entry)
        else:
            node_type = node.get("type")
            if isinstance(node_type, str) and node_type.lower() == "tool":
                extracted.append(
                    {
                        "name": node.get("name"),
                        "inputParameters": node.get("input")
                        or node.get("inputParameters"),
                        "output": node.get("output"),
                        "raw": node,
                    }
                )

    return extracted


def to_tool_call(entry: Dict[str, Any]) -> ToolCall:
    name = entry.get("name") or entry.get("toolName")
    input_parameters = (
        entry.get("inputParameters")
        or entry.get("input_parameters")
        or entry.get("input")
    )
    input_parameters = _parse_input_parameters(input_parameters)
    output = entry.get("output")
    description = entry.get("description")
    reasoning = entry.get("reasoning")
    return ToolCall(
        name=name or "unknown",
        description=description,
        reasoning=reasoning,
        output=output,
        input_parameters=input_parameters,
    )


def extract_tool_events_from_messages(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue

        raw_calls: List[Any] = []
        if "tool_calls" in message and isinstance(message["tool_calls"], list):
            raw_calls.extend(message["tool_calls"])

        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            raw_calls.append(function_call)

        for call in raw_calls:
            if not isinstance(call, dict):
                continue

            name = None
            input_parameters = None
            output = call.get("output")

            if "function" in call and isinstance(call["function"], dict):
                function_payload = call["function"]
                name = function_payload.get("name")
                arguments = function_payload.get("arguments")
            else:
                name = call.get("name")
                arguments = call.get("arguments")

            if isinstance(arguments, str):
                try:
                    input_parameters = json.loads(arguments)
                except json.JSONDecodeError:
                    input_parameters = arguments
            elif isinstance(arguments, dict):
                input_parameters = arguments

            extracted.append(
                {
                    "name": name,
                    "inputParameters": input_parameters,
                    "output": output,
                    "raw": call,
                }
            )

    return extracted


def build_expected_tool_calls(question: Dict[str, Any]) -> List[ToolCall]:
    expected_tools = question.get("expected_tools", []) or []
    tool_calls: List[ToolCall] = []
    for tool_name in expected_tools:
        params_source = None
        if tool_name == "parameter_scan":
            params_source = question.get("arguments", {})
        elif tool_name == "steady_state":
            params_source = question.get("arguments", {})
        elif tool_name == "search_models":
            params_source = {
                "query": question.get("search_query"),
                "number": question.get("expected_count"),
            }
        input_parameters = canonicalize_tool_input(
            tool_name, params_source, question, for_expected=True
        )
        tool_calls.append(
            ToolCall(
                name=tool_name,
                input_parameters=input_parameters,
            )
        )
    return tool_calls


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
    tools_called: List[Dict[str, Any]]


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
            normalize_message(msg) for msg in messages
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
            "thread_id": invocation_thread,
        }


t2b_runner = T2BAgentRunner(base_thread_prefix=thread_prefix, llm=llm_model)


# Cell 5 Execute Evaluation Loop
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

    thread_id = agent_payload["thread_id"]
    print(f"→ Thread {current_index:03d}: {thread_id} (question={question_id})")

    trace_obj = get_latest_trace()
    trace_tree = build_trace_dict(trace_obj)
    serialized_trace = dataclass_to_dict(trace_obj) if trace_obj else None
    tool_events = extract_tool_events_from_trace(trace_tree)
    if not tool_events:
        tool_events = extract_tool_events_from_messages(agent_payload["all_messages"])
    actual_tool_calls: List[ToolCall] = []
    for event in tool_events:
        call = to_tool_call(event)
        canonical_inputs = canonicalize_tool_input(
            call.name, call.input_parameters, question, for_expected=False
        )
        call.input_parameters = canonical_inputs
        actual_tool_calls.append(call)

    expected_tool_calls: List[ToolCall] = build_expected_tool_calls(question)

    metric_instance = ToolCorrectnessMetric(
        include_reason=True,
        threshold=0.5,
        verbose_mode=True,
        should_consider_ordering=True,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )

    test_case = LLMTestCase(
        input=question_text,
        actual_output=agent_payload["answer"],
        tools_called=actual_tool_calls,
        expected_tools=expected_tool_calls,
        additional_metadata={
            "model_id": question.get("model_id"),
            "expected_tools_raw": question.get("expected_tools"),
            "simulation_time": question.get("simulation_time"),
            "question_type": question.get("question_type"),
            "interval": question.get("interval"),
            "initial_concentration": question.get("initial_concentration"),
            "species": question.get("species"),
            "arguments": question.get("arguments"),
            "search_query": question.get("search_query"),
            "expected_count": question.get("expected_count"),
            "tool": question.get("tool"),
            "expected_values": question.get("expected_values"),
        },
        name=question_id,
    )

    score = metric_instance.measure(test_case)
    reason = metric_instance.reason or ""

    metric_scores.append(score)
    metric_reasons.append(reason)
    metric_details.append(
        {
            "question_id": question_id,
            "reason": reason,
            "verbose_logs": getattr(metric_instance, "verbose_logs", None),
            "question_type": question.get("question_type"),
            "actual_tool_calls": [
                call.model_dump(exclude_none=True)
                if hasattr(call, "model_dump")
                else {
                    "name": call.name,
                    "description": call.description,
                    "reasoning": call.reasoning,
                    "output": call.output,
                    "input_parameters": call.input_parameters,
                }
                for call in actual_tool_calls
            ],
            "expected_tool_calls": [
                call.model_dump(exclude_none=True)
                if hasattr(call, "model_dump")
                else {
                    "name": call.name,
                    "description": call.description,
                    "reasoning": call.reasoning,
                    "output": call.output,
                    "input_parameters": call.input_parameters,
                }
                for call in expected_tool_calls
            ],
        }
    )

    run_results.append(
        AgentRunResult(
            question_id=question_id,
            answer=agent_payload["answer"],
            assistant_messages=agent_payload["assistant_messages"],
            all_messages=agent_payload["all_messages"],
            state_fields=agent_payload["state_fields"],
            trace=serialized_trace,
            trace_tree=trace_tree,
            thread_id=thread_id,
            tools_called=[
                call.model_dump(exclude_none=True)
                if hasattr(call, "model_dump")
                else {
                    "name": call.name,
                    "description": call.description,
                    "reasoning": call.reasoning,
                    "output": call.output,
                    "input_parameters": call.input_parameters,
                }
                for call in actual_tool_calls
            ],
        )
    )

    print(f"[Tool Correctness] {question_id}: score={score:.3f}")

    reset_traces()


# Cell 6 Aggregate Results
average_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0

tool_correctness_summary = {
    "question_count": len(run_results),
    "average_score": average_score,
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
            "thread_id": result.thread_id,
            "tools_called": result.tools_called,
            "trace_available": result.trace is not None,
            "question_type": question_lookup.get(result.question_id, {}).get("question_type"),
            "interval": question_lookup.get(result.question_id, {}).get("interval"),
            "initial_concentration": question_lookup.get(result.question_id, {}).get("initial_concentration"),
            "expected_values": question_lookup.get(result.question_id, {}).get("expected_values"),
            "search_query": question_lookup.get(result.question_id, {}).get("search_query"),
            "expected_count": question_lookup.get(result.question_id, {}).get("expected_count"),
            "tool": question_lookup.get(result.question_id, {}).get("tool"),
            "expected_tools": [
                (
                    tc.model_dump(exclude_none=True)
                    if hasattr(tc, "model_dump")
                    else tc.__dict__
                )
                for tc in build_expected_tool_calls(question_lookup[result.question_id])
            ],
        }
        for idx, result in enumerate(run_results)
    ],
}

print("Tool Correctness evaluation complete.")
print(
    f"Evaluated {tool_correctness_summary['question_count']} questions. "
    f"Average score: {tool_correctness_summary['average_score']:.3f}"
)
print(f"Median score: {tool_correctness_summary['median_score']:.3f}")
print(
    f"Sample stdev: {tool_correctness_summary['stdev_sample']:.3f} | "
    f"Population stdev: {tool_correctness_summary['stdev_population']:.3f}"
)
print(
    f"Min score: {tool_correctness_summary['minimum_score']:.3f} | "
    f"Max score: {tool_correctness_summary['maximum_score']:.3f}"
)
print(
    f"Pass rate >=0.5: {tool_correctness_summary['pass_rate_threshold_0.5']:.3f} | "
    f"Pass rate >=0.7: {tool_correctness_summary['pass_rate_threshold_0.7']:.3f}"
)

output_payload = {
    "summary": tool_correctness_summary,
    "scores": metric_scores,
    "reasons": metric_reasons,
    "details": metric_details,
}

with TOOL_CORRECTNESS_OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)

print(f"Tool Correctness results saved to {TOOL_CORRECTNESS_OUTPUT_PATH}")
