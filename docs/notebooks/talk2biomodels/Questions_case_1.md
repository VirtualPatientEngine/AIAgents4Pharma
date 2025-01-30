# T2B sample use case #1
## Human disease model with drug treatment assessment: Crohns IL6 Disease

> Dwivedi G, Fitz L, Hegen M, Martin SW, Harrold J, Heatherington A, Li C. (2014). A multiscale model of interleukin-6-mediated immune regulation in Crohn's disease and its application in drug discovery and development. *CPT Pharmacometrics Syst Pharmacol*, 3(1):e89. doi: 10.1038/psp.2013.64. [PMID: 24402116](https://pubmed.ncbi.nlm.nih.gov/24402116/)

**Model**: [BIOMD0000000537](https://www.ebi.ac.uk/biomodels/BIOMD0000000537)

<hr>

### General questions about the model

<hr>

**Question 1:** Search for all models related to a disease/research term <br>
```Please search for all the BioModels on Crohn's Disease```

**Answer 1:** 
>| # | BioModel ID | BioModel Name | Format | Submission Date |
>|---|-------------|---------------|---------|-----------------|
>| 1 | BIOMD0000000535 | Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6 Antibody | SBML | 2014-08-04 |
>| 2 | BIOMD0000000536 | Dwivedi2014 - Crohns IL6 Disease model - sgp130 activity | SBML | 2014-08-04 |
>| 3 | BIOMD0000000537 | Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6R Antibody | SBML | 2014-08-04 |



**Q2:** Describe a model (the IL6 model for IBD) <br>
```Briefly describe the BioModel 537```

**A2:**

>**Description:**
>This model consists of four interconnected models that explore different aspects of Crohn's disease related to Interleukin-6 (IL-6) inhibition. The models include:
>
>- Healthy Volunteer model
>- Crohn's Disease - IL-6 Antibody
>- Crohn's Disease - sgp130FC
>- Crohn's Disease - IL-6Ra Antibody
>The primary focus is on simulating the differential activity of various ligands on IL-6 signaling, which impacts the activity of Signal Transducer and Activator of Transcription 3 (STAT3) and the production of the biomarker C-Reactive Protein (CRP).
>
>The model is detailed in the article titled "A multiscale model of interleukin-6-mediated immune regulation in Crohn's disease and its application in drug discovery and development," published in CPT Pharmacometrics & Systems Pharmacology in 2014.
>
>This model is hosted on the BioModels Database and can be identified by the ID BIOMD0000000537.

**Q3.** Ask about publication data of the model (It will remember the model number) <br>
```When and where was this model published?```

**A3:**

> The model was published in the article titled "A multiscale model of interleukin-6-mediated immune regulation in Crohn's disease and its application in drug discovery and development."

>**Publication Details:**
>
>Authors: Dwivedi G, Fitz L, Hegen M, Martin SW, Harrold J, Heatherington A, Li C.
>
>Journal: CPT Pharmacometrics & Systems Pharmacology
>
>Publication Date: 2014
>
>Volume: 3
>
>Article Identifier: e89
>
>Link to Article: PubMed
>
>This study discusses the development of a multiscale systems model for understanding IL-6-mediated immune regulation in Crohn's disease and its implications for drug discovery.

**Q4.** Display all parameters related to drug dosage. `Dose`=dosage every 4 weeks. `DoseQ2W`=dosage every 2 weeks. According to the article, by default `DoseQ2W` is
deactivated. <br>
```Show me the parameters related to drug dosing regimen in the model.```

**A4:** <br>

>The parameters related to the drug dosing regimen in the BioModel 537 include:
>
>- Dose: The amount of drug administered in a single dose.
>- DoseQ2W: The dosage administered every two weeks.
>- Initial for Dose: Initial concentration for the standard dose.
>- Initial for DoseQ2W: Initial concentration for the dosage administered every two weeks.
>- infusionTime: The duration of the drug infusion.

>These parameters are crucial for defining how the drug is administered over time and help in modeling the effects of different dosing schedules on the pharmacokinetics and pharmacodynamics of the drug.

<hr> 

### Interrogate model annotations
TBD (editing and saving annotations - work in progress)

### Model simulation and plotting

**Q5.** Simulate the model for 12 weeks (= 2016 hours), and give it a name. The current dose is set to 200 mg. <br>
```Simulate the model for 2016 hours with 2016 intervals. Set the initial concentration of Dose to 200 mg. Mark this simulation as `Treatment 4wk`.```



**Q6.** Show only CRP-related species, as it is one of the clinically-relevant measurement for a successful treatment. <br>
```Plot only CRP-related species.```



**Q7.** Review, how drug treatment influences CRP[serum] concentration at any arbitrary time point, e.g. 12 weeks after the initial drug treatment (2016 hours). <br>
```What is the CRP concentration in the serum after 2016 hours of treatment?```

<hr>

### Compare treatment regimens

**Q8.** Re-simulate the model by administrating the drug trieatment every two weeks by activating the `DoseQ2W` and deactivating `Dose` parameters. <br>  
```Simulate the model again for 2016 hours with 2016 intervals. Set the initial concentration of `DoseQ2W` to 200 and `Dose` to 0. Mark this simulation as `Treatment 2wk`.```


**Q9.** Compare the CRP values in the 2 cases. Treatment 2wk should reduce CRP over longer time than Treatment 4wk. <br>
```Based on the CRP values, which treatment would you recommend? `Treatment 4wk` or `Treatment 2wk`?```

<hr>

### Simulate the concentration of free drug in blood serum 

**Q10.** Show that the dose 500 mg helps maintain free anti–IL-6Rα at high concentration (Reproduce Figure 4f, 500 mg dose).
``` Simulate the model and set the initial concentration of Dose to 500 mg. Plot Ab in serum.```

<hr>

### Compare the effect on CRP suppression by simulating two different antibodies with varying dissociation affinities

**Q11.** Plot the current curve of CRP % suppression with the initial model parameters (`kIL6RUnbind` = 2.5). Compare the CRP suppression after 12 weeks of treatment. Dose is set to 500 mg. We expect 100% suppression of CRP after 12 weeks.
```Simulate the model and plot the curve of CRP % suppression. Save this simulation as `Antibody 2.5`.```

**Q12.** Set the parameter `kIL6RUnbind` to 250 (initial model value is 2.5), decreasing the stability of the antibody-IL6R complex. Simulate the model and plot serum CRP % suppression. We expect ~10% suppression of CRP after 12 weeks.
```Set the parameter `kIL6RUnbind` to 250 and simulate the model  for 2016 hours with 2016 intervals and plot the curve of CRP % suppression. Save this simulation as `Antibody 250`.```


**Q13.** Compare the two curves at 12 weeks. Which antibody `Antibody 250` or `Antibody 2.5` is better? The `Antibody 250` should have a higher CRP suppression after 12 weeks of treatment.
```Which antibody ('Antibody 250' or 'Antibody 2.5') is better, in terms of CRP suppression after 12 weeks of treatment?```

### Perform parameter scanning
**Q14.** Perform parameter scanning to see how the CRP in serum change if the parameter `kIL6RUnbind` is varied from 2.5 to 250.
```How does the CRP in serum change if the parameter `kIL6RUnbind` is varied from 2.5 to 250?```	