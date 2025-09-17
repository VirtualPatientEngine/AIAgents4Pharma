# Benchmark

We would like to benchmark the performance of the T2B agent.

Specifically, we would like to benchmarkt the following aspects of the T2B agent:

* stability of the outputs depending on stochastic user inputs and multiple sessions
* stability of the outputs given different prompts (short vs long, language etc.)

We'd like to test following aspects:
1. Stability of the outputs depending on the specific wording of the prompt (tools, like `simulate_model`, `ask_question`, `custom_plotter`, `model_description`, `search_models`, `fetch_parameters`, `get_annotation`)
    * Use the same prompt with different wording (e.g. different 'background' or 'context' of the user, also spelling errors and the length of the prompt).
        * Short vs long prompt (short to very long)
        * Context given very precise to very vague (experimental biologist's language vs computational biologist's language).
        * Spelling errors from few to a lot.
2. Input parameter correctness (Did the agent supply the correct parameters to the tools?) 

## Benchmakr strategy:

1. Generate a set of prompts that are representative of the user inputs and prompts that are used in the T2B agent. Use a placeholder values for arguments,such as, model id and parameter values.
    * Questions should be diverse with respect to lenght, background (immunologist, modeller, computational biologist, etc.) and complexity (short vs complex). Include grammatical errors and typos.

2. Generate a ground truth answer for every prompt.

3. Create and run a T2B agent for each prompt in a notebook.

