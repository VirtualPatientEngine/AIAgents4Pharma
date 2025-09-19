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

3. Create and run a T2B agent for each prompt in a notebook and evalate the precision with deepeval framework.


## Benchmark "User input context" 
**Aim**: test the stability of the outputs depending on the specific wording of the prompt. Keep parameters constant, vary only prompt quality and context.

**Data**: Diverse questions with respect to lenght, background (immunologist, modeller, computational biologist, etc.) and complexity (short vs complex). Include grammatical errors and typos. 50 questions per tool (Set 1).

**Models**: 537, 971, BIOMD0000000027 (use cases from the publication)

**Tools**: `simulate_model` (done), `ask_question` (done), `custom_plotter`, `model_description`, `search_models`, `fetch_parameters`, `get_annotation`, `parameter_scan`

**Tools to-do**: `parameter_scan`, `steady_state`, `search_models`

## Benchmark "Input parameter correctness" 
**Aim**: test the stability of the outputs depending on the specific parameters. Keep prompt quality and context constant, vary only parameters.

**Data**: Precise prompts with different parameters. Includes multiple models and parameter versions (Set 2).

**Tools**: `simulate_model`, `ask_question`, `custom_plotter`, `model_description`, `search_models`, `fetch_parameters`, `get_annotation`


## Benchmark "Tool selection correctness" 
Data: (Set 2)

Tools: `simulate_model`, `ask_question`, `custom_plotter`, `model_description`, `search_models`, `fetch_parameters`, `get_annotation`


# Prompt generation


**Simulate model**

*Simulate model tool Set 1 (constant parameters -vary language)*

 I would like to benchmark the agentic tool system that simulates biological ordinary differential equation questions. Generate 20 questions for a tool simulate_model, which has following description: ["A tool to simulate a biomodel"]. The questions should be diverse with respect to their lenght, user's background (immunologist would be vague in terms of name of specific operations, modeller would be precise in terms of tool names and expected output) and complexity (short, concisue questions vs casual vs complex and convoluted questions). Include grammatical errors and typos. Rate each generated question with a score between 0 and 10, from 0 easy to comprehend for the tool to 10 very complex and difficult to comprehend questions, add an id for every question. Add expected answer to the question as a text that would typically be returned by the tool based on LLM, but include expected answer, which can be rounded. Return the questions and the scores in a JSON format.

 Do not vary simulation time and species. 

You can include following variation with respect to the simulation conditions and model id:
* model id 537 - 12 weeks of simulation input in the prompt should be in hours - species CRP (c-reactive protein) in serum - expected answer 2.26913 nmol/L.

* model id 971 - 50 days of simulation with an interval of 50 - species infected covid patients - expected answer 104339.

* model id BIOMD0000000027 - 1000 seconds of simulation with an interval of 1000 - species Mpp (doubel posphorylated Mitogen-activated protein kinase 1) in the cell - expected answer 48.1723 nmol/L.

*Simulate model tool Set 2 (vary parameters: time interval and species)*

 I would like to benchmark the agentic tool system that simulates biological ordinary differential equation questions. Generate 30 questions for each modelid for a tool simulate_model, which has following description: ["A tool to simulate a biomodel"]. 

 The aim of these questions is to test the stability of the outputs depending on the specified input parameters. Keep prompt quality and context with a slight variation, vary only parameters.
 
 The questions should be somewhat diverse with respect to their lenght and complexity (short, concisue questions vs casual vs complex and convoluted questions). The questions should ask for the final concentration of the given species Add an id for every question. Add expected answer to the question as a text that would typically be returned by the tool based on LLM, but include expected answer, which can be rounded. Return the questions and the scores in a JSON format. 

 Now vary simulation time, interval and concentration of species to be simulated accodring to the instructions for each model id provided below. 

You can include following variation with respect to the simulation conditions and model id (see attached dictionary of model ids and parameters). Question should ask for the final concentration of the given species. Match the expected answer for each species with the key in the dictionary and the parameters that were used to generate the expected answer (interval, time, species name, initial concentration of this species):

Model id 537:
{'interval_2016_time_2016_species_CRP{serum}_concentration_0.01': CRP{serum}    10.301993
 CRP{liver}     6.707914
 IL6{serum}     0.000641
 STAT3{gut}     9.124980
 Name: 2016.0, dtype: float64,
 'interval_100_time_20_species_CRP{serum}_concentration_1000': CRP{serum}    243.477247
 CRP{liver}    162.273954
 IL6{serum}      0.003962
 STAT3{gut}      0.291445
 Name: 20.0, dtype: float64,
 'interval_4032_time_2016_species_CRP{serum}_concentration_2.6': CRP{serum}    10.302718
 CRP{liver}     6.708469
 IL6{serum}     0.000641
 STAT3{gut}     9.124957
 Name: 2016.0, dtype: float64,
 'interval_1000_time_1000_species_IL6{serum}_concentration_435628.8965511659': CRP{serum}    91.299723
 CRP{liver}    60.894795
 IL6{serum}     0.000642
 STAT3{gut}     6.813039
 Name: 1000.0, dtype: float64,
 'interval_1000_time_2016_species_CRP{liver}_concentration_1583.2584678161063': CRP{serum}    10.328675
 CRP{liver}     6.728273
 IL6{serum}     0.000641
 STAT3{gut}     9.124149
 Name: 2016.0, dtype: float64,
 'interval_500_time_500_species_STAT3{gut}_concentration_6.106360135082123e-08': CRP{serum}    215.468283
 CRP{liver}    154.750841
 IL6{serum}      0.000441
 STAT3{gut}      0.884170
 Name: 500.0, dtype: float64,
 'interval_2000_time_1500_species_STAT3{gut}_concentration_6.106360135082122e-10': CRP{serum}    159.484293
 CRP{liver}    110.490735
 IL6{serum}      0.000643
 STAT3{gut}      4.080634
 Name: 1500.0, dtype: float64}
 

 Model id 971 (vary only interval and time and output species name and respective value):
{'interval_100_time_50': Infected        1.043385e+05
 Susceptible     1.017891e+06
 Recovered       2.231583e+06
 Hospitalised    1.325140e+05
 Name: 50.0, dtype: float64,
 'interval_200_time_100': Infected        7.143353e+04
 Susceptible     1.055609e+06
 Recovered       4.586298e+06
 Hospitalised    8.688262e+04
 Name: 100.0, dtype: float64,
 'interval_400_time_180': Infected        4.031662e+04
 Susceptible     1.055585e+06
 Recovered       6.955857e+06
 Hospitalised    4.901975e+04
 Name: 180.0, dtype: float64,
 'interval_1000_time_500': Infected        4.090069e+03
 Susceptible     1.055585e+06
 Recovered       9.714202e+06
 Hospitalised    4.972991e+03
 Name: 500.0, dtype: float64,
 'interval_400_time_20': Infected        1.206681e+05
 Susceptible     3.167693e+06
 Recovered       1.634611e+05
 Hospitalised    4.076004e+04
 Name: 20.0, dtype: float64,
 'interval_10_time_10': Infected        3.413654e+03
 Susceptible     1.079620e+07
 Recovered       2.980390e+03
 Hospitalised    8.065950e+02
 Name: 10.0, dtype: float64}

Model id BIOMD0000000027 (vary only time and parameter name and value):
{'time_100_parameter_k1cat_value_0.1': Mpp    317.356267
 M      154.982325
 Mp      27.661408
 Name: 100.0, dtype: float64,
 'time_1000_parameter_k1cat_value_1': Mpp    494.565011
 M        0.076041
 Mp       5.358948
 Name: 1000.0, dtype: float64,
 'time_180_parameter_k1cat_value_100': Mpp    494.647985
 M        0.000758
 Mp       5.351257
 Name: 180.0, dtype: float64,
 'time_500_parameter_k2cat_value_10': Mpp    491.982136
 M        0.001140
 Mp       8.016723
 Name: 500.0, dtype: float64,
 'time_20_parameter_k2cat_value_0.1': Mpp     42.542954
 M        0.103107
 Mp     457.353939
 Name: 20.0, dtype: float64,
 'time_10_parameter_k2cat_value_100': Mpp    499.195565
 M        0.000113
 Mp       0.804322
 Name: 10.0, dtype: float64}

 ## Steady state and ask question
