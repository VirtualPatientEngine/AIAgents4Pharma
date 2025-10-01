# Benchmark

## Evaluation results

| Dataset | Task completion (mean ± std)  | 
|----------|----------|
| Set 1    | 0.712 ± 0.021   |
| Set 2    | 0.707 ± 0.028   |
| Set 3    | pending  | 
| Set 4    | pending  |


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
