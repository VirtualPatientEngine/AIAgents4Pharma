# Generating QnA pairs with a LLM and ground truth data

## Simulate model tool

*Simulate model tool Set 1 (constant parameters -vary language)*

 I would like to benchmark the agentic tool system that simulates biological ordinary differential equation questions. Generate 20 questions for a tool simulate_model, which has following description: ["A tool to simulate a biomodel"]. The questions should be diverse with respect to their lenght, user's background (immunologist would be vague in terms of name of specific operations, modeller would be precise in terms of tool names and expected output) and complexity (short, concisue questions vs casual vs complex and convoluted questions). Include grammatical errors and typos. Rate each generated question with a score between 0 and 10, from 0 easy to comprehend for the tool to 10 very complex and difficult to comprehend questions, add an id for every question. Add expected answer to the question as a text that would typically be returned by the tool based on LLM, but include expected answer, which can be rounded. Return the questions and the scores in a JSON format.

 Do not vary simulation time and species. 

You can include following variation with respect to the simulation conditions and model id:
* model id 537 - 12 weeks of simulation input in the prompt should be in hours - species CRP (c-reactive protein) in serum - expected answer 2.26913 nmol/L.

* model id 971 - 50 days of simulation with an interval of 50 - species infected covid patients - expected answer 104339.

* model id BIOMD0000000027 - 1000 seconds of simulation with an interval of 1000 - species Mpp (doubel posphorylated Mitogen-activated protein kinase 1) in the cell - expected answer 48.1723 nmol/L.

*Simulate model tool Set 2 (vary parameters: time interval and species)*

 I would like to benchmark the agentic tool system that simulates biological ordinary differential equation questions. Generate 30 questions for each model id for a tool simulate_model, which has following description: ["A tool to simulate a biomodel"]. 

 The aim of these questions is to test the stability of the outputs depending on the specified input parameters. Keep prompt quality and context with a slight variation, vary only parameters.
 
 The questions should be somewhat diverse with respect to their lenght and complexity (short, concisue questions vs casual vs complex and convoluted questions). The questions should ask for the final concentration of the given species Add an id for every question. Add expected answer to the question as a text that would typically be returned by the tool based on LLM, but include expected answer, which can be rounded. Return the questions and the scores in a JSON format. 

 Now vary simulation time, interval and concentration of species to be simulated accodring to the instructions for each model id provided below. 

You can include following variation with respect to the simulation conditions and model id (see attached dictionary of model ids and parameters). Question should ask for the final concentration of the given species. Match the expected answer for each species with the key in the dictionary and the parameters that were used to generate the expected answer (interval, time, species name, initial concentration of this species):

**Model id 537**
*(varying interval, time and species concentrations)*

Interval: 2016, Time: 2016, Initial CRP(serum) concentration: 0.01
| Species | Value |
|---------|-------|
| CRP(serum) | 10.301993 |
| CRP(liver) | 6.707914 |
| IL6(serum) | 0.000641 |
| STAT3(gut) | 9.124980 |

Interval: 100, Time: 20, Initial CRP(serum) concentration: 1000
| Species | Value |
|---------|-------|
| CRP(serum) | 243.477247 |
| CRP(liver) | 162.273954 |
| IL6(serum) | 0.003962 |
| STAT3(gut) | 0.291445 |

Interval: 4032, Time: 2016, Initial CRP(serum) concentration: 2.6
| Species | Value |
|---------|-------|
| CRP(serum) | 10.302718 |
| CRP(liver) | 6.708469 |
| IL6(serum) | 0.000641 |
| STAT3(gut) | 9.124957 |

Interval: 1000, Time: 1000, Initial IL6(serum) concentration: 435628.90
| Species | Value |
|---------|-------|
| CRP(serum) | 91.299723 |
| CRP(liver) | 60.894795 |
| IL6(serum) | 0.000642 |
| STAT3(gut) | 6.813039 |

Interval: 1000, Time: 2016, Initial CRP(liver) concentration: 1583.26
| Species | Value |
|---------|-------|
| CRP(serum) | 10.328675 |
| CRP(liver) | 6.728273 |
| IL6(serum) | 0.000641 |
| STAT3(gut) | 9.124149 |

Interval: 500, Time: 500, Initial STAT3(gut) concentration: 6.11e-08
| Species | Value |
|---------|-------|
| CRP(serum) | 215.468283 |
| CRP(liver) | 154.750841 |
| IL6(serum) | 0.000441 |
| STAT3(gut) | 0.884170 |

Interval: 2000, Time: 1500, Initial STAT3(gut) concentration: 6.11e-10
| Species | Value |
|---------|-------|
| CRP(serum) | 159.484293 |
| CRP(liver) | 110.490735 |
| IL6(serum) | 0.000643 |
| STAT3(gut) | 4.080634 |
 

**Model id 971**
*(varying interval and time)*

Interval: 100, Time: 50
| Species | Value |
|---------|-------|
| Infected | 1.043385e+05 |
| Susceptible | 1.017891e+06 |
| Recovered | 2.231583e+06 |
| Hospitalised | 1.325140e+05 |

Interval: 200, Time: 100
| Species | Value |
|---------|-------|
| Infected | 7.143353e+04 |
| Susceptible | 1.055609e+06 |
| Recovered | 4.586298e+06 |
| Hospitalised | 8.688262e+04 |

Interval: 400, Time: 180
| Species | Value |
|---------|-------|
| Infected | 4.031662e+04 |
| Susceptible | 1.055585e+06 |
| Recovered | 6.955857e+06 |
| Hospitalised | 4.901975e+04 |

Interval: 1000, Time: 500
| Species | Value |
|---------|-------|
| Infected | 4.090069e+03 |
| Susceptible | 1.055585e+06 |
| Recovered | 9.714202e+06 |
| Hospitalised | 4.972991e+03 |

Interval: 400, Time: 20
| Species | Value |
|---------|-------|
| Infected | 1.206681e+05 |
| Susceptible | 3.167693e+06 |
| Recovered | 1.634611e+05 |
| Hospitalised | 4.076004e+04 |

Interval: 10, Time: 10
| Species | Value |
|---------|-------|
| Infected | 3.413654e+03 |
| Susceptible | 1.079620e+07 |
| Recovered | 2.980390e+03 |
| Hospitalised | 8.065950e+02 |

**Model id BIOMD0000000027**
*(varying time and parameter name/value)*

Time: 100s, Parameter: k1cat = 0.1
| Species | Concentration |
|---------|--------------|
| Mpp | 317.356267 |
| M | 154.982325 |
| Mp | 27.661408 |

Time: 1000s, Parameter: k1cat = 1.0
| Species | Concentration |
|---------|--------------|
| Mpp | 494.565011 |
| M | 0.076041 |
| Mp | 5.358948 |

Time: 180s, Parameter: k1cat = 100
| Species | Concentration |
|---------|--------------|
| Mpp | 494.647985 |
| M | 0.000758 |
| Mp | 5.351257 |

Time: 500s, Parameter: k2cat = 10
| Species | Concentration |
|---------|--------------|
| Mpp | 491.982136 |
| M | 0.001140 |
| Mp | 8.016723 |

Time: 20s, Parameter: k2cat = 0.1
| Species | Concentration |
|---------|--------------|
| Mpp | 42.542954 |
| M | 0.103107 |
| Mp | 457.353939 |

Time: 10s, Parameter: k2cat = 100
| Species | Concentration |
|---------|--------------|
| Mpp | 499.195565 |
| M | 0.000113 |
| Mp | 0.804322 |

## Steady state tool

 I would like to benchmark the agentic tool system that simulates biological ordinary differential equation models. I am benchamrking a tool steady_state, which brings a model to steady state. 

 The aim of these questions is to test the stability of the outputs depending on the specified input parameters. Keep prompt quality and context with a slight variation, vary only parameters.
 

  Generate 30 questions for each model id for a steady state and ask question tools. Add generated questions to the set 2.

  Example question: "Bring biomodel 27 to a steady state, and then determine the Mpp concentration at the steady state."

  ASk a question if the model reaches steady state and what is the steady state concentration for the model species written in the table below. Add an id for every question. Add expected answer to the question as a text that would typically be returned by the tool based on LLM, but include expected answer, which can be rounded. The answers should be obtained from the table below. Return the questions and the scores in a JSON format.

For model 537, steady state cannot be found. 

**Model 27:**
| Name | Concentration [nmol/l] | Transition Time [s] |
|------|----------------------|-------------------|
| M | 0.076041 | 1.012 |
| Mp | 5.358948 | 0.669 |
| Mpp | 494.565011 | 62.278 |

**Model 971 (covid model):**
| Name | Concentration [#/ml] | Transition Time [d] |
|------|---------------------|-------------------|
| Susceptible | 1055585.406125 | 1.376e+27 |
| Exposed | -5.661e-22 | 7.000 |
| Infected | -1.541e-22 | 2.160 |
| Asymptomatic | -8.028e-23 | 7.154 |
| Susceptible_quarantined | -1.074e-20 | 14.000 |
| Exposed_quarantined | -1.222e-28 | 7.943 |
| Hospitalised | -1.873e-22 | 8.602 |
| Recovered | 10025626.335858 | 1.195e+29 |

## Search models tool

 I would like to benchmark the agentic tool system that searches for models in the BioModels database. Generate 20 questions for a tool search_models, which has following description: ["A tool to search for curated models in the BioModels database"]. The questions should be diverse with respect to their lenght, and complexity (short, concisue questions vs casual vs complex and convoluted questions). Include grammatical errors and typos. Rate each generated question with a score between 0 and 10, from 0 easy to comprehend for the tool to 10 very complex and difficult to comprehend questions, add an id for every question. Add expected answer to the question as a text that would typically be returned by the tool based on LLM, but include expected answer, which can be rounded. Return the questions and the scores in a JSON format.

 The questions should ask for the number of models found for the given query. The query should be diverse with respect to the length, complexity and context.

Example question: "Search for models on precision medicine, and then list the names of the models.", "how many models on crohn's disease are there in biomodels database?"

Use these keywords: " crohn's disease; precision medicine, covid-19, gut.

Precision medicine: 7 models
[{'format': 'SBML',
  'id': 'BIOMD0000000583',
  'lastModified': None,
  'name': 'Leber2015 - Mucosal immunity and gut microbiome interaction during C. difficile infection',
  'submissionDate': '2015-07-19T23:00:00Z',
  'submitter': 'Andrew Leber',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000583'},
 {'format': 'SBML',
  'id': 'BIOMD0000000962',
  'lastModified': None,
  'name': 'Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan, Hubei, and China',
  'submissionDate': '2020-08-25T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000962'},
 {'format': 'SBML',
  'id': 'MODEL2106070001',
  'lastModified': None,
  'name': 'Montagud2022 - Prostate cancer Boolean model',
  'submissionDate': '2021-06-06T23:00:00Z',
  'submitter': 'Arnau Montagud',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2106070001'},
 {'format': 'SBML',
  'id': 'MODEL2203300001',
  'lastModified': None,
  'name': 'Bannerman2022 Whole Genome Metabolism - Mycobacterium leprae (strain Br4923)_1',
  'submissionDate': '2022-03-29T23:00:00Z',
  'submitter': 'Bridget Bannerman',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2203300001'},
 {'format': 'SBML',
  'id': 'MODEL2203300002',
  'lastModified': None,
  'name': 'Bannerman2022 Whole Genome Metabolism - M. abscessus (strain ATCC 19977 / DSM 44196)',
  'submissionDate': '2022-03-29T23:00:00Z',
  'submitter': 'Bridget Bannerman',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2203300002'},
 {'format': 'SBML',
  'id': 'MODEL2402290001',
  'lastModified': None,
  'name': 'Gulhane2022 - Sphingolipid metabolism and PI3K-AKT axis in NSCLC',
  'submissionDate': '2024-02-29T00:00:00Z',
  'submitter': 'Shailza Singh',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2402290001'},
 {'format': 'Open Neural Network Exchange',
  'id': 'MODEL2407230001',
  'lastModified': None,
  'name': 'YangziChen2024 - Metabolic Machine Learning Predictor Model for Diagnosis of Gastric Cancer',
  'submissionDate': '2024-07-22T23:00:00Z',
  'submitter': 'Akshat Pandey',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2407230001'}]

  crohn's disease: 5 models
  [{'format': 'SBML',
  'id': 'BIOMD0000000534',
  'lastModified': None,
  'name': 'Dwivedi2014 - Healthy Volunteer IL6 Model',
  'submissionDate': '2014-08-04T23:00:00Z',
  'submitter': 'Vincent Knight-Schrijver',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000534'},
 {'format': 'SBML',
  'id': 'BIOMD0000000535',
  'lastModified': None,
  'name': 'Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6 Antibody',
  'submissionDate': '2014-08-04T23:00:00Z',
  'submitter': 'Vincent Knight-Schrijver',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000535'},
 {'format': 'SBML',
  'id': 'BIOMD0000000536',
  'lastModified': None,
  'name': 'Dwivedi2014 - Crohns IL6 Disease model - sgp130 activity',
  'submissionDate': '2014-08-04T23:00:00Z',
  'submitter': 'Vincent Knight-Schrijver',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000536'},
 {'format': 'SBML',
  'id': 'BIOMD0000000537',
  'lastModified': None,
  'name': 'Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6R Antibody',
  'submissionDate': '2014-08-04T23:00:00Z',
  'submitter': 'Vincent Knight-Schrijver',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000537'},
 {'format': 'SBML',
  'id': 'MODEL2505090001',
  'lastModified': None,
  'name': 'Lo2016 – IBD Response to Anti-TNFα treatment',
  'submissionDate': '2025-05-08T23:00:00Z',
  'submitter': 'bastien chassagnol',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2505090001'}]

  covid-19: 43 models
  [{'format': 'SBML',
  'id': 'BIOMD0000000955',
  'lastModified': None,
  'name': 'Giordano2020 - SIDARTHE model of COVID-19 spread in Italy',
  'submissionDate': '2020-07-27T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000955'},
 {'format': 'SBML',
  'id': 'BIOMD0000000956',
  'lastModified': None,
  'name': 'Bertozzi2020 - SIR model of scenarios of COVID-19 spread in CA and NY',
  'submissionDate': '2020-08-06T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000956'},
 {'format': 'SBML',
  'id': 'BIOMD0000000957',
  'lastModified': None,
  'name': 'Roda2020 - SIR model of COVID-19 spread in Wuhan',
  'submissionDate': '2020-08-10T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000957'},
 {'format': 'SBML',
  'id': 'BIOMD0000000958',
  'lastModified': None,
  'name': 'Ndairou2020 - early-stage transmission dynamics of COVID-19 in Wuhan',
  'submissionDate': '2020-08-13T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000958'},
 {'format': 'SBML',
  'id': 'BIOMD0000000960',
  'lastModified': None,
  'name': 'Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19',
  'submissionDate': '2020-08-19T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000960'},
 {'format': 'SBML',
  'id': 'BIOMD0000000962',
  'lastModified': None,
  'name': 'Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan, Hubei, and China',
  'submissionDate': '2020-08-25T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000962'},
 {'format': 'SBML',
  'id': 'BIOMD0000000963',
  'lastModified': None,
  'name': 'Weitz2020 - SIR model of COVID-19 transmission with shielding',
  'submissionDate': '2020-09-15T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000963'},
 {'format': 'SBML',
  'id': 'BIOMD0000000964',
  'lastModified': None,
  'name': 'Mwalili2020 - SEIR model of COVID-19 transmission and environmental pathogen prevalence',
  'submissionDate': '2020-09-20T23:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000964'},
 {'format': 'SBML',
  'id': 'BIOMD0000000969',
  'lastModified': None,
  'name': 'Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio',
  'submissionDate': '2020-10-27T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000969'},
 {'format': 'SBML',
  'id': 'BIOMD0000000970',
  'lastModified': None,
  'name': 'Hou2020 - SEIR model of COVID-19 transmission in Wuhan',
  'submissionDate': '2020-10-28T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000970'},
 {'format': 'SBML',
  'id': 'BIOMD0000000971',
  'lastModified': None,
  'name': 'Tang2020 - Estimation of transmission risk of COVID-19 and impact of public health interventions',
  'submissionDate': '2020-11-02T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000971'},
 {'format': 'SBML',
  'id': 'BIOMD0000000972',
  'lastModified': None,
  'name': 'Tang2020 - Estimation of transmission risk of COVID-19 and impact of public health interventions - update',
  'submissionDate': '2020-11-03T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000972'},
 {'format': 'SBML',
  'id': 'BIOMD0000000974',
  'lastModified': None,
  'name': 'Carcione2020 - Deterministic SEIR simulation of a COVID-19 outbreak',
  'submissionDate': '2020-11-03T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000974'},
 {'format': 'SBML',
  'id': 'BIOMD0000000976',
  'lastModified': None,
  'name': 'Ghanbari2020 - forecasting the second wave of COVID-19 in Iran',
  'submissionDate': '2020-11-23T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000976'},
 {'format': 'SBML',
  'id': 'BIOMD0000000977',
  'lastModified': None,
  'name': 'Sarkar2020 - SAIR model of COVID-19 transmission with quarantine measures in India',
  'submissionDate': '2020-11-30T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000977'},
 {'format': 'SBML',
  'id': 'BIOMD0000000978',
  'lastModified': None,
  'name': 'Mukandavire2020 - SEIR model of early COVID-19 transmission in South Africa',
  'submissionDate': '2020-12-01T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000978'},
 {'format': 'SBML',
  'id': 'BIOMD0000000979',
  'lastModified': None,
  'name': 'Malkov2020 - SEIRS model of COVID-19 transmission with reinfection',
  'submissionDate': '2020-12-04T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000979'},
 {'format': 'SBML',
  'id': 'BIOMD0000000980',
  'lastModified': None,
  'name': 'Malkov2020 - SEIRS model of COVID-19 transmission with time-varying R values and reinfection',
  'submissionDate': '2020-12-07T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000980'},
 {'format': 'SBML',
  'id': 'BIOMD0000000981',
  'lastModified': None,
  'name': 'Wan2020 - risk estimation and prediction of the transmission of COVID-19 in maninland China excluding Hubei province',
  'submissionDate': '2020-12-09T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000981'},
 {'format': 'SBML',
  'id': 'BIOMD0000000982',
  'lastModified': None,
  'name': 'Law2020 - SIR model of COVID-19 transmission in Malyasia with time-varying parameters',
  'submissionDate': '2020-12-24T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000982'},
 {'format': 'SBML',
  'id': 'BIOMD0000000983',
  'lastModified': None,
  'name': 'Zongo2020 - model of COVID-19 transmission dynamics under containment measures in France',
  'submissionDate': '2021-01-15T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000983'},
 {'format': 'SBML',
  'id': 'BIOMD0000000984',
  'lastModified': None,
  'name': 'Fang2020 - SEIR model of COVID-19 transmission considering government interventions in Wuhan',
  'submissionDate': '2021-01-19T00:00:00Z',
  'submitter': 'Kausthubh Ramachandran',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000984'},
 {'format': 'SBML',
  'id': 'BIOMD0000000988',
  'lastModified': None,
  'name': 'Westerhoff2020 - systems biology model of the coronavirus pandemic 2020',
  'submissionDate': '2021-02-12T00:00:00Z',
  'submitter': 'Paul Jonas Jost',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000988'},
 {'format': 'SBML',
  'id': 'BIOMD0000000991',
  'lastModified': None,
  'name': 'Okuonghae2020 - SEAIR model of COVID-19 transmission in Lagos, Nigeria',
  'submissionDate': '2021-02-18T00:00:00Z',
  'submitter': 'Paul Jonas Jost',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000991'},
 {'format': 'COMBINE archive',
  'id': 'MODEL2003020001',
  'lastModified': None,
  'name': 'Renz2020 - GEM of Human alveolar macrophage with SARS-CoV-2',
  'submissionDate': '2020-03-02T00:00:00Z',
  'submitter': 'Andreas Dräger',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2003020001'},
 {'format': 'SBML',
  'id': 'MODEL2007210001',
  'lastModified': None,
  'name': 'Bannerman2020 - Integrated model of the human airway epithelial cell and the SARS-CoV-2 virus',
  'submissionDate': '2020-07-20T23:00:00Z',
  'submitter': 'Bridget Bannerman',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2007210001'},
 {'format': 'SBML',
  'id': 'MODEL2109130003',
  'lastModified': None,
  'name': 'COVID-19 immunotherapy A mathematical model',
  'submissionDate': '2022-06-20T23:00:00Z',
  'submitter': 'João Nuno Domingues Tavares',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2109130003'},
 {'format': 'SBML',
  'id': 'MODEL2111170001',
  'lastModified': None,
  'name': 'Leander2021 - innate immune response to SARS-CoV-2 in the alveolar epithelium',
  'submissionDate': '2021-11-17T00:00:00Z',
  'submitter': 'Krishna Kumar Tiwari',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2111170001'},
 {'format': 'SBML',
  'id': 'MODEL2202230002',
  'lastModified': None,
  'name': 'COVID-19 immunotherapy A mathematical model',
  'submissionDate': '2022-02-23T00:00:00Z',
  'submitter': 'João Nuno Domingues Tavares',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2202230002'},
 {'format': 'COMBINE archive',
  'id': 'MODEL2202240001',
  'lastModified': None,
  'name': 'New workflow predicts drug targets against SARS-CoV-2 via metabolic changes in infected cells',
  'submissionDate': '2023-01-18T00:00:00Z',
  'submitter': 'Andreas Dräger',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2202240001'},
 {'format': 'SBML',
  'id': 'MODEL2203300001',
  'lastModified': None,
  'name': 'Bannerman2022 Whole Genome Metabolism - Mycobacterium leprae (strain Br4923)_1',
  'submissionDate': '2022-03-29T23:00:00Z',
  'submitter': 'Bridget Bannerman',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2203300001'},
 {'format': 'SBML',
  'id': 'MODEL2203300002',
  'lastModified': None,
  'name': 'Bannerman2022 Whole Genome Metabolism - M. abscessus (strain ATCC 19977 / DSM 44196)',
  'submissionDate': '2022-03-29T23:00:00Z',
  'submitter': 'Bridget Bannerman',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2203300002'},
 {'format': 'MATLAB (Octave)',
  'id': 'MODEL2205010001',
  'lastModified': None,
  'name': 'PraneshPadmanabhan2022 - SARS-CoV-2 virus dynamics model (human)',
  'submissionDate': '2022-04-30T23:00:00Z',
  'submitter': 'Rajat Desikan',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2205010001'},
 {'format': 'Other',
  'id': 'MODEL2209020001',
  'lastModified': None,
  'name': 'BY-COVID Knowledge Graph: A comprehensive network integrating various data resources of BY-COVID project',
  'submissionDate': '2022-09-01T23:00:00Z',
  'submitter': 'Reagon Karki',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2209020001'},
 {'format': 'Other',
  'id': 'MODEL2212160001',
  'lastModified': None,
  'name': 'Gianlupi2022 - Antiviral Timing, Potency, and Heterogeneity Effects on an Epithelial Tissue Patch Infected by SARS-CoV-2',
  'submissionDate': '2022-12-16T00:00:00Z',
  'submitter': 'Juliano Ferrari Gianlupi',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2212160001'},
 {'format': 'SBML',
  'id': 'MODEL2212310001',
  'lastModified': None,
  'name': 'Model-based prediction of SARS-CoV-2 in Ethiopia: Extended SEIR model',
  'submissionDate': '2022-12-31T00:00:00Z',
  'submitter': 'Simon Merkt',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2212310001'},
 {'format': 'Other',
  'id': 'MODEL2401170001',
  'lastModified': None,
  'name': 'Pradeep2024 - A Mechanical modelling framework for studying endothelial permeability',
  'submissionDate': '2024-01-17T00:00:00Z',
  'submitter': 'Pradeep Keshavanarayana',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2401170001'},
 {'format': 'Python',
  'id': 'MODEL2405080005',
  'lastModified': None,
  'name': 'Zeng2022 - Prediction of molecular properties and drug targets using a self-supervised learning framework',
  'submissionDate': '2024-05-07T23:00:00Z',
  'submitter': 'Zainab Ashimiyu-Abdusalam',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2405080005'},
 {'format': 'Python',
  'id': 'MODEL2405080006',
  'lastModified': None,
  'name': 'Stokes2020 - Antibiotics discovery using deep learning approach (Antiviral implementation)',
  'submissionDate': '2024-05-07T23:00:00Z',
  'submitter': 'Zainab Ashimiyu-Abdusalam',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2405080006'},
 {'format': 'Python',
  'id': 'MODEL2405130001',
  'lastModified': None,
  'name': 'Zeng2022 - Prediction of HIV Growth Inhibition using ImageMol',
  'submissionDate': '2024-05-12T23:00:00Z',
  'submitter': 'Zainab Ashimiyu-Abdusalam',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2405130001'},
 {'format': 'Python',
  'id': 'MODEL2405130004',
  'lastModified': None,
  'name': 'KC2021 - A machine learning platform to estimate anti-SARS-CoV-2 activities',
  'submissionDate': '2024-05-12T23:00:00Z',
  'submitter': 'Zainab Ashimiyu-Abdusalam',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2405130004'},
 {'format': 'COMBINE archive',
  'id': 'MODEL2503040001',
  'lastModified': None,
  'name': 'Miroshnichenko2025 - A modular model of immune response to SARS-CoV-2 infection',
  'submissionDate': '2025-03-04T00:00:00Z',
  'submitter': 'Max',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2503040001'},
 {'format': 'SBML',
  'id': 'MODEL2504120001',
  'lastModified': None,
  'name': 'Merkt2024 - SEIR based model of SARS-CoV-2 variant spread and cross-immunity in Ethiopia',
  'submissionDate': '2025-04-11T23:00:00Z',
  'submitter': 'Simon Merkt',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2504120001'}]

  gut: 58 models
[{'format': 'SBML',
  'id': 'BIOMD0000000451',
  'lastModified': None,
  'name': 'Carbo2013 - Cytokine driven CD4+ T Cell differentiation and phenotype plasticity',
  'submissionDate': '2013-04-22T23:00:00Z',
  'submitter': 'Adria Carbo',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000451'},
 {'format': 'SBML',
  'id': 'BIOMD0000000527',
  'lastModified': None,
  'name': 'Kaiser2014 - Salmonella persistence after ciprofloxacin treatment',
  'submissionDate': '2013-12-17T00:00:00Z',
  'submitter': 'Roland Regoes',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000527'},
 {'format': 'SBML',
  'id': 'BIOMD0000000583',
  'lastModified': None,
  'name': 'Leber2015 - Mucosal immunity and gut microbiome interaction during C. difficile infection',
  'submissionDate': '2015-07-19T23:00:00Z',
  'submitter': 'Andrew Leber',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000583'},
 {'format': 'SBML',
  'id': 'BIOMD0000000619',
  'lastModified': None,
  'name': 'Sluka2016  - Acetaminophen PBPK',
  'submissionDate': '2015-09-22T23:00:00Z',
  'submitter': 'Vijayalakshmi Chelliah',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000619'},
 {'format': 'SBML',
  'id': 'BIOMD0000000947',
  'lastModified': None,
  'name': 'Lee2017 - Paracetamol first-pass metabolism PK model',
  'submissionDate': '2018-03-05T00:00:00Z',
  'submitter': 'Matthew Roberts',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000947'},
 {'format': 'SBML',
  'id': 'BIOMD0000001062',
  'lastModified': None,
  'name': 'Kim2021 - Development of a Genome-Scale Metabolic Model and Phenome Analysis of the Probiotic Escherichia coli Strain Nissle 1917',
  'submissionDate': '2022-04-17T23:00:00Z',
  'submitter': 'Dohyeon Kim',
  'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000001062'},
 {'format': 'SBML',
  'id': 'MODEL1210260004',
  'lastModified': None,
  'name': 'Viladomiu2012 - PPARgamma role in C.diff associated disease',
  'submissionDate': '2012-10-25T23:00:00Z',
  'submitter': 'Monica Viladomiu',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1210260004'},
 {'format': 'SBML',
  'id': 'MODEL1509220000',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Brain medulla',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220000'},
 {'format': 'SBML',
  'id': 'MODEL1509220001',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Embryonic tissue',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220001'},
 {'format': 'SBML',
  'id': 'MODEL1509220002',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Cerebellum',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220002'},
 {'format': 'SBML',
  'id': 'MODEL1509220003',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Colon',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220003'},
 {'format': 'SBML',
  'id': 'MODEL1509220004',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Eye',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220004'},
 {'format': 'SBML',
  'id': 'MODEL1509220005',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Diaphragm',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220005'},
 {'format': 'SBML',
  'id': 'MODEL1509220006',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Brown fat',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220006'},
 {'format': 'SBML',
  'id': 'MODEL1509220007',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Jejunum',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220007'},
 {'format': 'SBML',
  'id': 'MODEL1509220008',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Kidney medulla',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220008'},
 {'format': 'SBML',
  'id': 'MODEL1509220009',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Ileum',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220009'},
 {'format': 'SBML',
  'id': 'MODEL1509220010',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Duodenum',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220010'},
 {'format': 'SBML',
  'id': 'MODEL1509220011',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Adrenal gland',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220011'},
 {'format': 'SBML',
  'id': 'MODEL1509220012',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Kidney cortex',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220012'},
 {'format': 'SBML',
  'id': 'MODEL1509220013',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Brain cortex',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220013'},
 {'format': 'SBML',
  'id': 'MODEL1509220014',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Heart',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220014'},
 {'format': 'SBML',
  'id': 'MODEL1509220015',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Lung',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220015'},
 {'format': 'SBML',
  'id': 'MODEL1509220016',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Muscle',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220016'},
 {'format': 'SBML',
  'id': 'MODEL1509220017',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Liver',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220017'},
 {'format': 'SBML',
  'id': 'MODEL1509220018',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Midbrain',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220018'},
 {'format': 'SBML',
  'id': 'MODEL1509220019',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Thymus',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220019'},
 {'format': 'SBML',
  'id': 'MODEL1509220020',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Spleeen',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220020'},
 {'format': 'SBML',
  'id': 'MODEL1509220021',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Stomach',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220021'},
 {'format': 'SBML',
  'id': 'MODEL1509220022',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Salivary gland',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220022'},
 {'format': 'SBML',
  'id': 'MODEL1509220023',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Ovary',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220023'},
 {'format': 'SBML',
  'id': 'MODEL1509220024',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Pancreas',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220024'},
 {'format': 'SBML',
  'id': 'MODEL1509220025',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Olfactory bulb',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220025'},
 {'format': 'SBML',
  'id': 'MODEL1509220026',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - White fat',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220026'},
 {'format': 'SBML',
  'id': 'MODEL1509220027',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Tissue-specific genome-scale metabolic network - Uterus',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220027'},
 {'format': 'SBML',
  'id': 'MODEL1509220028',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Generic mouse genome-scale metabolic network (MMR)',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220028'},
 {'format': 'SBML',
  'id': 'MODEL1509220029',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Curated tissue-specific genome-scale metabolic network - Liver',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220029'},
 {'format': 'SBML',
  'id': 'MODEL1509220030',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Curated tissue-specific genome-scale metabolic network  - Colon',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220030'},
 {'format': 'SBML',
  'id': 'MODEL1509220031',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Curated tissue-specific genome-scale metabolic network - Adipose',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220031'},
 {'format': 'SBML',
  'id': 'MODEL1509220032',
  'lastModified': None,
  'name': 'Mardinoglu2015 - Curated tissue-specific genome-scale metabolic model - Small intestine',
  'submissionDate': '2015-09-21T23:00:00Z',
  'submitter': 'Nicolas Rodriguez',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL1509220032'},
 {'format': 'SBML',
  'id': 'MODEL2002040002',
  'lastModified': None,
  'name': 'Ankrah2021 - Genome scale metabolic model of Drosophila gut microbe Acetobacter fabarum',
  'submissionDate': '2020-02-04T00:00:00Z',
  'submitter': 'Nana Y D Ankrah',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2002040002'},
 {'format': 'SBML',
  'id': 'MODEL2002040003',
  'lastModified': None,
  'name': 'Ankrah2021 - Genome scale metabolic model of Drosophila gut microbe Acetobacter pomorum',
  'submissionDate': '2020-02-04T00:00:00Z',
  'submitter': 'Nana Y D Ankrah',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2002040003'},
 {'format': 'SBML',
  'id': 'MODEL2002040004',
  'lastModified': None,
  'name': 'Ankrah2021 - Genome scale metabolic model of Drosophila gut microbe Acetobacter tropicalis',
  'submissionDate': '2020-02-04T00:00:00Z',
  'submitter': 'Nana Y D Ankrah',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2002040004'},
 {'format': 'SBML',
  'id': 'MODEL2002040005',
  'lastModified': None,
  'name': 'Ankrah2021 - Genome scale metabolic model of Drosophila gut microbe Lactobacillus brevis',
  'submissionDate': '2020-02-04T00:00:00Z',
  'submitter': 'Nana Y D Ankrah',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2002040005'},
 {'format': 'SBML',
  'id': 'MODEL2002040006',
  'lastModified': None,
  'name': 'Ankrah2021 - Genome scale metabolic model of Drosophila gut microbe Lactobacillus plantarum',
  'submissionDate': '2020-02-04T00:00:00Z',
  'submitter': 'Nana Y D Ankrah',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2002040006'},
 {'format': 'SBML',
  'id': 'MODEL2002070001',
  'lastModified': None,
  'name': 'Geißert2020 - Yersinia enterocolitica co-infection in mice',
  'submissionDate': '2020-02-07T00:00:00Z',
  'submitter': 'Andreas Dräger',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2002070001'},
 {'format': 'SBML',
  'id': 'MODEL2010130002',
  'lastModified': None,
  'name': 'Subrmanian2015 - Energy metabolism of Leishmania infantum, constrain-based metabolic model',
  'submissionDate': '2020-10-12T23:00:00Z',
  'submitter': 'Abhishek Subramanian',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2010130002'},
 {'format': 'SBML',
  'id': 'MODEL2110210002',
  'lastModified': None,
  'name': 'MirhakkakSchaeuble2021 - a Candida albicans genome-scale metabolic model reconstruction',
  'submissionDate': '2021-10-20T23:00:00Z',
  'submitter': 'Sascha Schäuble',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2110210002'},
 {'format': 'SBML',
  'id': 'MODEL2210190001',
  'lastModified': None,
  'name': 'Koduru2022 - Lactobacillus casei subsp. casei ATCC 393',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190001'},
 {'format': 'SBML',
  'id': 'MODEL2210190002',
  'lastModified': None,
  'name': 'Koduru2022 - Lactobacillus casei BL23',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190002'},
 {'format': 'SBML',
  'id': 'MODEL2210190003',
  'lastModified': None,
  'name': 'Koduru2022 - Lactobacillus casei LC5',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190003'},
 {'format': 'SBML',
  'id': 'MODEL2210190004',
  'lastModified': None,
  'name': 'Koduru2022 - Lactobacillus fermentum ATCC 14931',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190004'},
 {'format': 'SBML',
  'id': 'MODEL2210190006',
  'lastModified': None,
  'name': 'Koduru2022 - Lactobacillus plantarum ATCC8014',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190006'},
 {'format': 'SBML',
  'id': 'MODEL2210190007',
  'lastModified': None,
  'name': 'Koduru2022 - Lactobacillus plantarum JDM1',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190007'},
 {'format': 'SBML',
  'id': 'MODEL2210190008',
  'lastModified': None,
  'name': 'Koduru2022 - Lactobacillus salivarius ATCC 11741',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190008'},
 {'format': 'SBML',
  'id': 'MODEL2210190009',
  'lastModified': None,
  'name': 'Koduru2022 - Lactococcus lactis subsp. cremoris NZ9000',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190009'},
 {'format': 'SBML',
  'id': 'MODEL2210190012',
  'lastModified': None,
  'name': 'Koduru2022 - Leuconostoc mesenteroides subsp. mesenteroides ATCC 8293',
  'submissionDate': '2022-10-18T23:00:00Z',
  'submitter': 'Yi Qing Lee',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2210190012'},
 {'format': 'Other',
  'id': 'MODEL2405300001',
  'lastModified': None,
  'name': 'Ioannou2024 - Synthetic Human Infant Gut Microbial Community with Human Milk Oligosaccharides',
  'submissionDate': '2024-05-29T23:00:00Z',
  'submitter': 'William Scott',
  'url': 'https://www.ebi.ac.uk/biomodels/MODEL2405300001'}]

## Custom plotter tool
Create a set of questions to test a tool custom_plotter (A visualization tool designed to extract and display a subset
of the larger simulation plot generated by the simulate_model tool.
It allows users to specify particular species for the y-axis,
providing a more targeted view of key species without the clutter
of the full plot.).

Run time course and plot only speceis that are specified in the question.The evaluation framework will check if the certain species are plotted in the plot that is returned by the agent.

Model species in model 537, e.g, IL6{serum}, CRP{serum}, sR{serum}, CRP{liver} and STAT3{liver}.

Model species in model 27, e.g, Mpp, M, Mp.

Model species in model 971, e.g, Infected, Susceptible, Recovered, Hospitalised, Exposed.

Create a questions that ask to simulate the respective model first (allways ask to simulate the model first) and plot the species that are specified in the question (mix species).

The key "species" should contain the species names that are plotted in the plot.

Create diverse 20 questions for each model id 537, 27 and 971 and add them to the set 2 in a similar fashion as are the questions in the set2.

The example question and expected answer in json format is here:


    {
      "id": "custom_plotter_set2_001",
      "question": "simulate the model 537 and plot the species IL6{serum}, CRP{serum}, sR{serum}, CRP{liver} and STAT3{liver}",
      "complexity_score": 2,
      "user_background": "modeller",
      "model_id": "537",
      "question_type": "custom_plotter",
      "expected_values": {
        "species": "IL6{serum}, CRP{serum}, sR{serum}, CRP{liver}, STAT3{liver}"
      },
      "expected_tools": [
        "simulate_model",
        "custom_plotter"
      ]
    }

## Get model info tool
Create a questions for set3 in a similar fashion as are the questions in the set2 for the tool `get_modelinfo`.
Extract detailed information about the model, including species, parameters, and compartments.

Template questions: 
* "What are the species do you have in the model 971?
* "what is the time unit in the model 27?"

Ask the question about these three different models and include expected answer to the question based on the template below:

Questions regarding **units** (please generate 10 questions for each model id 27, 537 and 971) can ask about time unit, quantity unit, length unit, area unit and volume unit.

Questions regarding **compartments** (please generate 10 questions for each model id 27, 537 and 971) can ask about the compartment name and size in the model.

Questions regarding **species** (please generate 10 questions for each model id 27, 537 and 971) can ask about the species name, compartment, type, unit, initial concentration and display name.

Questions regarding **parameters** (please generate 10 questions for each model id 27, 537 and 971) can ask about the parameter name, type, unit, initial value and display name.

  The ground truths for each question is in the tables or pickle files in the `tables` folder, depending on whether the quesiton is asked about units, compartments, species or parameters.

  The example question and expected answer in json format is here:

      {
      "id": "get_modelinfo_set3_001",
      "question": "what are the species do you have in the model 971?",
      "complexity_score": 2,
      "user_background": "modeller",
      "model_id": "971",
      "question_type": "table",
      "expected_values": {
        "species": "tables/species_971.csv"
      },
      "expected_tools": [
        "get_modelinfo"
      ]
    }


## Steady state tool
Create a questions for set3 in a similar fashion as are the questions in the set2 for the tool `steady_state`.

Questions regarding **steady state** (please generate 10 questions for each model id 27 and 971) can ask about the steady state concentration and transition time of the species in the model.

The ground truths for each question is in the tables in the `tables` folder, depending on model id.

The example question and expected answer in json format is here:

    {
      "id": "steady_state_set3_001", 
      "question": "what is the steady state concentration of species M in model 27?",
      "complexity_score": 2,
      "user_background": "modeller",
      "model_id": "27",
      "question_type": "steady_state",
      "expected_values": {
        "steady_state": "tables/stst_27.csv"
      },
      "expected_tools": [
        "steady_state"
      ]
    }



## Get annotation tool

Create a questions for set4 in a similar fashion as are the questions in the set2 for the tool `get_annotation`.
This tool returns the annotation ID, if user asks for a specific species or annotations for all the species in the model.

Create 20 questions for each model id 27, 537 and 971 and add them to the set 4 in a similar fashion as are the questions in the set2 and set3. Make sure the questions are diverse and include different species and different number of species. Ask specificallly for the annotations of the species. Expected values should be the annotation ID, one species can return multiple annotation IDs. Expected tools should be get_annotation.

Use these ground truths for the questions and answer generation:

Model id 27
M
{'descriptions': [{'id': 'P26696', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P26696', 'resource': 'UniProt Knowledgebase'}]}
Mp
{'descriptions': [{'id': 'P26696', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P26696', 'resource': 'UniProt Knowledgebase'}]}
Mpp
{'descriptions': [{'id': 'P26696', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P26696', 'resource': 'UniProt Knowledgebase'}]}
MAPKK
{'descriptions': [{'id': 'Q05116', 'qualifier': 'is', 'uri': 'http://identifiers.org/uniprot/Q05116', 'resource': 'UniProt Knowledgebase'}]}
MKP3
{'descriptions': [{'id': 'Q90W58', 'qualifier': 'is', 'uri': 'http://identifiers.org/uniprot/Q90W58', 'resource': 'UniProt Knowledgebase'}]}

Model id 537:
IL6{serum}
{'descriptions': [{'id': 'P05231', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}]}
sgp130{serum}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
sR_IL6_sgp130{serum}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'has part', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
CRP{serum}
{'descriptions': [{'id': 'P02741', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P02741', 'resource': 'UniProt Knowledgebase'}]}
sR{serum}
{'descriptions': [{'id': 'P08887', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
sR_IL6{serum}
{'descriptions': [{'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
Ab{serum}
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}, {'id': 'CHEBI:52217', 'qualifier': 'has property', 'uri': 'http://identifiers.org/CHEBI:52217', 'resource': 'ChEBI'}]}
Ab_sR{serum}
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'has part', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
Ab_sR_IL6{serum}
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'has part', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
CRP Suppression (%)
{'descriptions': [{'id': 'P02741', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P02741', 'resource': 'UniProt Knowledgebase'}]}
CRP (% of baseline)
{'descriptions': [{'id': 'P02741', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P02741', 'resource': 'UniProt Knowledgebase'}]}
gp130{liver}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}]}
R_IL6_gp130{liver}
{'descriptions': [{'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'PR:000001868', 'qualifier': 'has part', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}]}
sR_IL6{liver}
{'descriptions': [{'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
R
{'descriptions': [{'id': 'P08887', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}]}
IL6{liver}
{'descriptions': [{'id': 'P05231', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}]}
R_IL6
{'descriptions': [{'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}]}
Ractive{liver}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'has part', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0002354', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0002354', 'resource': 'PATO'}, {'id': 'SBO:0000286', 'qualifier': 'has property', 'uri': 'http://identifiers.org/SBO:0000286', 'resource': 'Systems Biology Ontology'}]}
STAT3{liver}
{'descriptions': [{'id': 'PR:000002089', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000002089', 'resource': 'Protein Ontology'}]}
pSTAT3{liver}
{'descriptions': [{'id': 'PR:000002089', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000002089', 'resource': 'Protein Ontology'}, {'id': 'SBO:0000216', 'qualifier': 'has property', 'uri': 'http://identifiers.org/SBO:0000216', 'resource': 'Systems Biology Ontology'}]}
CRP{liver}
{'descriptions': [{'id': 'P02741', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P02741', 'resource': 'UniProt Knowledgebase'}]}
sR{liver}
{'descriptions': [{'id': 'P08887', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
CRPExtracellular
{'descriptions': [{'id': 'P02741', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P02741', 'resource': 'UniProt Knowledgebase'}]}
sgp130{liver}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
sR_IL6_sgp130{liver}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'has part', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
Ab_sR{liver}
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'has part', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
Ab{liver}
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}]}
Ab_R
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'has part', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}]}
Ab_sR_IL6{liver}
{'descriptions': [{'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'FMA:62871', 'qualifier': 'has part', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
sR_IL6{gut}
{'descriptions': [{'id': 'P05231', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
gp130{gut}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}]}
R_IL6_gp130{gut}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'has part', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}]}
Ractive{gut}
{'descriptions': [{'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'PR:000001868', 'qualifier': 'has part', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0002354', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0002354', 'resource': 'PATO'}, {'id': 'SBO:0000286', 'qualifier': 'has property', 'uri': 'http://identifiers.org/SBO:0000286', 'resource': 'Systems Biology Ontology'}]}
STAT3{gut}
{'descriptions': [{'id': 'PR:000002089', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000002089', 'resource': 'Protein Ontology'}]}
pSTAT3{gut}
{'descriptions': [{'id': 'PR:000002089', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000002089', 'resource': 'Protein Ontology'}, {'id': 'SBO:0000216', 'qualifier': 'has property', 'uri': 'http://identifiers.org/SBO:0000216', 'resource': 'Systems Biology Ontology'}]}
geneProduct
{'descriptions': [{'id': 'CCO:U0000003', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/CCO:U0000003', 'resource': 'Cell Cycle Ontology'}]}
sR{gut}
{'descriptions': [{'id': 'P08887', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
IL6{gut}
{'descriptions': [{'id': 'P05231', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}]}
sgp130{gut}
{'descriptions': [{'id': 'PR:000001868', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
sR_IL6_sgp130{gut}
{'descriptions': [{'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PR:000001868', 'qualifier': 'has part', 'uri': 'http://identifiers.org/PR:000001868', 'resource': 'Protein Ontology'}, {'id': 'P05231', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
Ab{gut}
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}]}
Ab_sR{gut}
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'has part', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}, {'id': 'P08887', 'qualifier': 'has part', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}]}
Ab_sR_IL6{gut}
{'descriptions': [{'id': 'P05231', 'qualifier': 'has property', 'uri': 'http://identifiers.org/uniprot/P05231', 'resource': 'UniProt Knowledgebase'}, {'id': 'P08887', 'qualifier': 'has property', 'uri': 'http://identifiers.org/uniprot/P08887', 'resource': 'UniProt Knowledgebase'}, {'id': 'PATO:0001537', 'qualifier': 'has property', 'uri': 'http://identifiers.org/PATO:0001537', 'resource': 'PATO'}, {'id': 'FMA:62871', 'qualifier': 'has property', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}]}
Ab{peripheral}
{'descriptions': [{'id': 'FMA:62871', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/FMA:62871', 'resource': 'FMA'}]}

Model id 971:

Susceptible
{'descriptions': [{'id': '0000514', 'qualifier': 'is', 'uri': 'http://identifiers.org/ido/0000514', 'resource': 'Infectious Disease Ontology'}, {'id': 'C171133', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/ncit/C171133', 'resource': 'NCIt'}]}
Exposed
{'descriptions': [{'id': '0000514', 'qualifier': 'is', 'uri': 'http://identifiers.org/ido/0000514', 'resource': 'Infectious Disease Ontology'}, {'id': 'C171133', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/ncit/C171133', 'resource': 'NCIt'}, {'id': '0000597', 'qualifier': 'has property', 'uri': 'http://identifiers.org/ido/0000597', 'resource': 'Infectious Disease Ontology'}]}
Asymptomatic
{'descriptions': [{'id': '0000569', 'qualifier': 'is', 'uri': 'http://identifiers.org/ido/0000569', 'resource': 'Infectious Disease Ontology'}, {'id': '0000511', 'qualifier': 'is', 'uri': 'http://identifiers.org/ido/0000511', 'resource': 'Infectious Disease Ontology'}, {'id': 'C171133', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/ncit/C171133', 'resource': 'NCIt'}]}
Susceptible_quarantined
{'descriptions': [{'id': 'C71902', 'qualifier': 'is', 'uri': 'http://identifiers.org/ncit/C71902', 'resource': 'NCIt'}, {'id': '0000514', 'qualifier': 'is', 'uri': 'http://identifiers.org/ido/0000514', 'resource': 'Infectious Disease Ontology'}, {'id': 'C171133', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/ncit/C171133', 'resource': 'NCIt'}]}
Exposed_quarantined
{'descriptions': [{'id': '0000514', 'qualifier': 'is', 'uri': 'http://identifiers.org/ido/0000514', 'resource': 'Infectious Disease Ontology'}, {'id': 'C71902', 'qualifier': 'is', 'uri': 'http://identifiers.org/ncit/C71902', 'resource': 'NCIt'}, {'id': 'C171133', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/ncit/C171133', 'resource': 'NCIt'}, {'id': '0000597', 'qualifier': 'has property', 'uri': 'http://identifiers.org/ido/0000597', 'resource': 'Infectious Disease Ontology'}]}
Hospitalised
{'descriptions': [{'id': '0000511', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/ido/0000511', 'resource': 'Infectious Disease Ontology'}, {'id': 'C171133', 'qualifier': 'is version of', 'uri': 'http://identifiers.org/ncit/C171133', 'resource': 'NCIt'}, {'id': 'C25179', 'qualifier': 'has property', 'uri': 'http://identifiers.org/ncit/C25179', 'resource': 'NCIt'}]}
Recovered
{'descriptions': [{'id': '0000621', 'qualifier': 'has property', 'uri': 'http://identifiers.org/ido/0000621', 'resource': 'Infectious Disease Ontology'}]}

Example question in json format:
    {
      "id": "annotation_set4_001",
      "question": "get the annotation for the species IL6{liver} in model 537",
      "complexity_score": 2,
      "user_background": "modeller",
      "model_id": "537",
      "question_type": "get_annotation",
      "expected_values": {
        "IL6{liver}": "P05231"
      },
      "expected_tools": [
        "get_annotation"
      ]
    }

Example question in json format:
    {
      "id": "annotation_set4_001",
      "question": "get the annotation for the species IL6{liver} and pSTAT3{liver} in model 537",
      "complexity_score": 2,
      "user_background": "modeller",
      "model_id": "537",
      "question_type": "get_annotation",
      "expected_values": {
        "IL6{liver}": ["P05231"],
        "pSTAT3{liver}": ["PR:000002089",SBO:0000216]
      },
      "expected_tools": [
        "get_annotation"
      ]
    }

## Parameter scan tool

Create a questions for set4 in a similar fashion as are the questions in the set2 for the tool `parameter_scan`.
This tool returns the parameter scan results for the model.

Create 20 questions for each model id 27, 537 and 971 and add them to the set 5 in a similar fashion as are the questions in the set2 and set4. Make sure the questions are diverse and include different parameters and different number of parameters. Ask specifically for the parameter scan results. Expected values should be the parameter scan results in a csv file. Expected tools should be parameter_scan.

Use the following combination to mix and match the questions with answers:

EXAMPLE:
**Model 537**
Question 1: Run a param scan in model 537 to observe change in concentration of CRP in serum over time if initial value of the species IL6{gut} is changed from 2.101100e-08 to 2.101100e+02 with a step size of 42.02.

name = 'IL6{gut}'
output = 'CRP{serum}'
start_scan = 2.101100e-08
stop_scan = 2.101100e+02
stepsize_scan = (stop_scan-start_scan)/5.
filename = './tables/scan_IL6_CRP_537.csv'

Example in JSON format:

    {
      "id": "scan_IL6_CRP_537_001",
      "question": "Run a param scan in model 537 to observe change in concentration of CRP in serum over time if initial value of the species IL6{gut} is changed from 2.101100e-08 to 2.101100e+02 with a step size of 42.02.",
      "complexity_score": 2,
      "user_background": "modeller",
      "model_id": "537",
      "question_type": "parameter_scan",
      "arguments": {
        "species_names": ["IL6{gut}"],
        "species_output": "CRP{serum}",
        "species_parameter_values": [2.101100e-08, 2.101100e+02],
        "stepsize_scan": 42.02,
        "time": 100,
        "intervals": 10,
        "filename": "tables/scan_IL6_CRP_537.csv"
      },
      "expected_values": {
        "parameter_scan": "tables/scan_IL6_CRP_537.csv"
      },
      "expected_tools": [
        "parameter_scan"
      ]
    }

Combination parameters for **model 537**:

* sca1
name = 'R'
output = 'CRPExtracellular'
start_scan = 0.0043
stop_scan = 8.0043
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_R_CRPExtracellular_537.csv'

* scan2
name = 'R'
output = 'R_IL6'
start_scan = 0.0043
stop_scan = 8.0043
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_R_IL6_537.csv' 

* scan3
name = 'kdegIL6Gut'
output = 'R_IL6'
start_scan = 1e-3
stop_scan = 10
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_kdegIL6Gut_R_537.csv'


**Model 971**

* scan1
name = 'Exposed'
output = 'Recovered'
start_scan = 1
stop_scan = 10000
stepsize_scan = (stop_scan-start_scan)/5.
time = 100
intervals = 10
filename = './tables/scan_Exposed_Recovered_971.csv'

* scan2
name = 'beta'
output = 'Hospitalised'
start_scan = 2.101100e-08
stop_scan = 2.101100e-05
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_beta_Hospitalised_971.csv'

* scan3
name = 'beta'
output = 'Infected'
start_scan = 2.101100e-08
stop_scan = 2.101100e-05
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_beta_Infected_971.csv'

* scan4
name = 'sigma'
output = 'Hospitalised'
start_scan = 1.428570e-03
stop_scan = 1.428570
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_sigma_Hospitalised_971.csv'

* scan5
name = 'sigma'
output = 'Infected'
start_scan = 1.428570e-03
stop_scan = 1.428570
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_sigma_Infected_971.csv'


**Model 27**

* scan1
name = 'k1cat'
output = 'Mpp'
start_scan = 0.01
stop_scan = 1
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
print(stepsize_scan)
filename = './tables/scan_k1cat_Mpp_27.csv'

* scan2
name = 'k1cat'
output = 'M'
start_scan = 0.01
stop_scan = 1
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_k1cat_M_27.csv'

* scan3
name = 'k2cat'
output = 'Mpp'
start_scan = 0.01
stop_scan = 1
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_k2cat_Mpp_27.csv'

* scan4
name = 'k1cat'
output = 'M'
start_scan = 0.01
stop_scan = 1
stepsize_scan = (stop_scan-start_scan)/4.
time = 100
intervals = 10
filename = './tables/scan_k1cat_M_27.csv'

* scan5
name = 'MAPKK'
output = 'Mpp'
start_scan = 1
stop_scan = 100
stepsize_scan = (stop_scan-start_scan)/10.
time = 100
intervals = 10
filename = './tables/scan_MAPKK_Mpp_27.csv'


