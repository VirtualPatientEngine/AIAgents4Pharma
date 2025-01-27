# T2B sample use case #1
## Human disease model with drug treatment assessment: Crohns IL6 Disease

> Dwivedi G, Fitz L, Hegen M, Martin SW, Harrold J, Heatherington A, Li C. (2014). A multiscale model of interleukin-6-mediated immune regulation in Crohn's disease and its application in drug discovery and development. *CPT Pharmacometrics Syst Pharmacol*, 3(1):e89. doi: 10.1038/psp.2013.64. [PMID: 24402116](https://pubmed.ncbi.nlm.nih.gov/24402116/)

**Model**: [BIOMD0000000537](https://www.ebi.ac.uk/biomodels/BIOMD0000000537)

<hr>

### General questions about the model

**Question 1:** Search for all models related to a disease/research term <br>
```Please search for all the biomodels on Crohns Disease```

**Answer 1:** 
>| # | BioModel ID | BioModel Name | Format | Submission Date |
>|---|-------------|---------------|---------|-----------------|
>| 1 | BIOMD0000000535 | Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6 Antibody | SBML | 2014-08-04 |
>| 2 | BIOMD0000000536 | Dwivedi2014 - Crohns IL6 Disease model - sgp130 activity | SBML | 2014-08-04 |
>| 3 | BIOMD0000000537 | Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6R Antibody | SBML | 2014-08-04 |



**Q2:** Describe a model (the IL6 model for IBD) <br>
```Briefly describe the biomodel 537```

**A2:**

> BioModel 537 (Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6R Antibody) presents a multiscale systems model examining interleukin-6 (IL-6)-mediated immune regulation in Crohn's disease. The model integrates intracellular signaling with organ-level pharmacological marker dynamics and evaluates different biotherapeutic strategies targeting IL-6 signaling. It specifically analyzes ligand activity effects on IL-6 signaling and downstream impacts on STAT3 activity and CRP expression. Key findings suggest dual targeting of the IL-6/sIL-6R complex plus IL-6 or IL-6R provides superior suppression of Crohn's disease pharmacological markers versus alternative approaches (see Figures 3a, 4d, 4e, 4f, and 5b). Source: BioModels Database, ID: 537.

**Q3.** Ask about publication data of the model (It will remember the model number) <br>
```When and where was the biomodel published?```

**A3:**

> The biomodel 537 was published on January 8, 2014.

**Q4.** Display all parameters related to drug dosage. `Dose`=dosage every 4 weeks. `DoseQ2W`=dosage every 2 weeks. According to the article, by default `DoseQ2W` is
deactivated. <br>
```Show me the parameters related to drug dosage in the model.```

**A4:** <br>
> The parameters related to drug dosage in the model include:
> - Dose
> - DoseQ2W (Dose every two weeks)
> - Initial for CRP
> - Initial for DoseQ2W
> - Initial for Dose

<hr> 

### Model simulation and plotting

**Q5.** Simulate the model for 12 weeks (= 2016 hours), and give it a name. The current dose is set to 200 mg. <br>
```Simulate the model for 2016 hours and interval of 2016 hours. Set the initial concentration of Dose to 200 mg. Mark this simulation and its follow-up questions as `Treatment 4wk` in your memory.```



**Q6.** Show only CRP-related species, as it is one of the clinically-relevant measurement for a successful treatment. <br>
```Plot only CRP-related species```



**Q7.** Show CRP[serum] concentration at any arbitrary time point, e.g. 12 weeks after the initial drug treatment (2016 hours) <br>
```What is the CRP concentration in the serum after 12 weeks of treatment?```

<hr>

### Compare treatment regimens

**Q8.** Resimulate the model by activating the `DoseQ2W` and deactivating `Dose` parameters <br>  
```Simulate the model again for 2016 hours and intervals 2016 with an initial concentration of `DoseQ2W` set to 200 and `Dose` set to 0. Mark this simulation and its follow-up questions as `Treatment 2wk` in your memory.```


**A8:**

**Q11.** Compare the CRP values in the 2 cases. Treatment 2wk should reduce CRP over longer time than Treatment 4wk. <br>
```Based on the CRP values, which treatment would you recommend? `Treatment 4wk` or `Treatment 2wk`?```

<hr>

### Simulate the concentration of free drug in blood serum 

 Q. Show that the dose 500 mg helps maintain free anti–IL-6Rα at high concentration (Reproduce Figure 4f, 500 mg dose)
``` Simulate the model and set the initial concentration of Dose to 500 mg. Plot Ab in serum.```

<hr>

### Compare two different antibody affinities / test antibodies with different target binding affinities

**Q12** Plot the current curve of CRP % suppression with the initial model parameters (`kIL6RUnbind` = 2.5). Compare the CRP suppression after 12 weeks of treatment. Dose is set to 500 mg. We expect 100% suppression of CRP after 12 weeks.
```Simulate the model and plot the curve of CRP % suppression. Save this simulation as `Antibody 2.5`.```

**Q12.** Set the parameter `kIL6RUnbind` to 250 (initial model value is 2.5), decreasing the stability of the antibody-IL6R complex. Simulate the model and plot serum CRP % supression. We expect ~10% suppression of CRP after 12 weeks.
```Set the parameter `kIL6RUnbind` to 250 and simulate the model  for 2016 hours with 2016 intervals and plot the curve of CRP % suppression. Save this simulation as `Antibody 250`.```

--- not plotting

**Q13.** Compare the two curves at 12 weeks. Which antibody `Antibody 250` or `Antibody 2.5` is better? 
```Which antibody affinity is better?```

--- memory does not work