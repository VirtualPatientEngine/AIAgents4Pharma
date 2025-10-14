# Benchmark

## Evaluation results
| Dataset | Task completion (mean ± std)  | 
|----------|----------|
| Set 1    | 0.712 ± 0.021   |
| Set 2    | 0.707 ± 0.028   |
| Set 3    | pending  | 
| Set 4    | pending  |

For detailed results, please refer to log files for [set 1](results/task_completion_set1_results.json) and [set 2](results/task_completion_set2_results.json).

### Example evaluation

"id": "sim_090" (set 1)
Question: "I need to understand the MAPK signaling dynamics for my research. Please simulate model BIOMD0000000027 for 1000 seconds and provide the Mpp concentration at the end of the simulation period. This information is critical for my research on signal transduction and cellular responses."
Expected answer: "I've simulated model BIOMD0000000027 for 1000 seconds to analyze the MAPK signaling dynamics for your research. The Mpp concentration at the end of the simulation period is 48.17 nmol/L. This information is critical for your research on signal transduction and cellular responses, as it represents the steady-state concentration of doubly phosphorylated MAPK1 after the complete 1000-second simulation."

Answer during benchmark: "The concentration of Mpp at the end of the 1000 seconds simulation is approximately 48.18 nmol. This information should be useful for your research on signal transduction and cellular responses.",

Task completion score: 0.7
LLM-as-a-judge verdict: "The system successfully processed the input data and provided an answer, which aligns with the task of answering a question. However, the inclusion of additional keys such as 'assistant_messages', 'all_messages', 'state_fields', and 'thread_id' suggests that the output was more complex than necessary for the task. The lack of additional tools or handoffs indicates a straightforward approach, but the extra information may not have been directly relevant to answering the question."



## Description
We would like to benchmark the performance on *Task Completion* of the T2B agent using [DeepEval framework](https://deepeval.com/docs/getting-started). Here, T2B's generated response is evaluated against the ground truth answer using a LLM-as-a-judge.

Specifically, we would like to benchmarkt the following aspects of the T2B agent:

* stability of the outputs depending on stochastic user inputs, grammatical errors, typos, length of the prompt and user background
* stability of the outputs given different tool calls, argument inputs and multi-turn conversation


## Dataset summary

| Set | Description | Tools | Focus | Questions | Example |
|-----|-------------|-------|-------|-----------|---------|
| [Set 1](benchmark_questions_set1.json) | User input variability with respect to background, grammar and clarity. Captures extreme cases in how users can address the agent. | simulate_model, ask_question | Communication variability | 90 | "pls run sim BIOMD0000000027 1000 seconds get Mpp concentration" vs "I need to understand the MAPK signaling dynamics for my research..." |
| [Set 2](benchmark_questions_set2.json) | Variability in user inputs relative to the number of provided parameters and tools, requested through generally well-formulated and grammatically correct questions. | simulate_model, search_models, steady_state, ask_question, custom_plotter, get_modelinfo | Parameter variability | 222 | "Search for models on precision medicine, and then list the names of the models." |
| [Set 3](benchmark_questions_set3.json) | Tabular data matching | steady_state, parameter_scan | Tabular data | 79 | "Analyze MAPKK parameter impact on Mpp concentration over time in model 27. Use parameter scan from 1 to 100 with 10 steps." |
| [Set 4](benchmark_questions_set4.json) | Annotation id matching | get_annotation | Annotation matching | 60 | "what are the annotations for Mp and MKP3 in model 27?" |

## Benchmark strategy

1. Generate set of prompts and ground truth answers for each set. The ground truth answers are generated using the [basico library](https://github.com/copasi/basico) and textualized using a LLM. Each set should represent a different output types (textual, tabular, dictionary, etc.) and different tool calling patterns (single parameter, multiple parameters, multiple tools, etc.). The prompts that were used to generate the ground truth answers can be found [here](generating_QnA_pairs.md) and the ground truth data can be found [here](expected_asnwers_basico.ipynb).

2. Runn Task Completion benchmark for each set. 

The code used for the benchmark can be found here for [set 1](results/Task_Completion_set1.py) and [set 2](results/Task_Completion_set2.py).
