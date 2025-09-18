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

## Benchmark strategy:

1. Generate a set of prompts that are representative of the user inputs and prompts that are used in the T2B agent. Use a placeholder values for arguments,such as, model id and parameter values.
    * Questions should be diverse with respect to lenght, background (immunologist, modeller, computational biologist, etc.) and complexity (short vs complex). Include grammatical errors and typos.

2. Generate a ground truth answer for every prompt.
    * genereate the results from basico
    * generate textual answers providing the question and groudn truth from basico.

3. Create and run a T2B agent for each prompt in a notebook.

## Benchmark "User input context" 
**Aim**: test the stability of the outputs depending on the specific wording of the prompt.

**Data**: Diverse questions with respect to lenght, background (immunologist, modeller, computational biologist, etc.) and complexity (short vs complex). Include grammatical errors and typos. 50 questions per tool (Set 1).

**Tools**: `simulate_model`, `ask_question`, `custom_plotter`, `model_description`, `search_models`, `fetch_parameters`, `get_annotation`

## Benchmark "Input parameter correctness" 
**Data**: Precise prompts with different parameters. Includes multiple models and parameter versions (Set 2).

**Tools**: `simulate_model`, `ask_question`, `custom_plotter`, `model_description`, `search_models`, `fetch_parameters`, `get_annotation`


## Benchmark "Tool selection correctness" 
Data: (Set 2)

Tools: `simulate_model`, `ask_question`, `custom_plotter`, `model_description`, `search_models`, `fetch_parameters`, `get_annotation`


# Prompt generation

Use this prompt to generate the prompts for the benchmarks:

Set 1:

I would like to benchmakr the agentic tool system that simulates biological ordinary differential equation questions. Generate 10 questions for a tool [tool_name], which has following description: [tool_description]. The questions should be diverse with respect to their lenght, user's background (immunologist would be vague in terms of name of specific operations, modeller would be precise in terms of tool names and expected output) and complexity (short, concisue questions vs complex and convoluted questions). Include grammatical errors and typos. Rate each generated question with a score between 0 and 10, from 0 easy to comprehend for the tool to 10 very complex and difficult to comprehend questions, add an id for every question. Return the questions and the scores in a JSON format. The example of the questions (clear) is "what is the concentration of c-reactive protein in the serum after 10 weeks of simulation when simulating model 537?" You may vary simulation time and model id from these examples 971 and BIOMD0000000027.


**Simulate model**

I would like to benchmark the agentic tool system that simulates biological ordinary differential equation questions. Generate 10 questions for a tool simulate_model, which has following description: ["A tool to simulate a biomodel"]. The questions should be diverse with respect to their lenght, user's background (immunologist would be vague in terms of name of specific operations, modeller would be precise in terms of tool names and expected output) and complexity (short, concisue questions vs complex and convoluted questions). Include grammatical errors and typos. Rate each generated question with a score between 0 and 10, from 0 easy to comprehend for the tool to 10 very complex and difficult to comprehend questions, add an id for every question. Add expected answer to the question. Return the questions and the scores in a JSON format.
You can include following variation with respect to the simulation conditions and model id:
* model id 537 - up to 15 weeks of simulation - species CRP (c-reactive protein) in serum, IL6R (interleuking 6 receptor) in liver gut or serum, and STAT (signal transducer and activator of transcription) in liver or gut.
* model id 971 - up to 50 days of simulation with an interval of 50 - species succeptible, infected and hospitalized covid patients.
* model id BIOMD0000000027 - up to 100 hours of simulation with an interval of 100 - species M, Mp (phosphorylated M) and MAPKK (map kinase kinase) in the cell.
 The example of the questions is "what is the concentration of c-reactive protein in the serum after 10 weeks of simulation when simulating model 537?" 

 I would like to benchmark the agentic tool system that simulates biological ordinary differential equation questions. Generate 20 questions for a tool simulate_model, which has following description: ["A tool to simulate a biomodel"]. The questions should be diverse with respect to their lenght, user's background (immunologist would be vague in terms of name of specific operations, modeller would be precise in terms of tool names and expected output) and complexity (short, concisue questions vs complex and convoluted questions). Include grammatical errors and typos. Rate each generated question with a score between 0 and 10, from 0 easy to comprehend for the tool to 10 very complex and difficult to comprehend questions, add an id for every question. Add expected answer to the question as a text that would typically be returned by the tool based on LLM. Return the questions and the scores in a JSON format.

 Do not vary simulation time and species. 

You can include following variation with respect to the simulation conditions and model id:
* model id 537 - 12 weeks of simulation input in the prompt should be in hours - species CRP (c-reactive protein) in serum - expected answer 2.26913 nmol/L.

* model id 971 - 50 days of simulation with an interval of 50 - species infected covid patients - expected answer 104339.

* model id BIOMD0000000027 - 1000 seconds of simulation with an interval of 1000 - species Mpp (doubel posphorylated Mitogen-activated protein kinase 1) in the cell - expected answer 48.1723 nmol/L.

