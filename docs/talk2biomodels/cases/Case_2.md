# Example Use Case #2
## Predicting effectiveness of contact inhibition during SARS-CoV-2 virus pandemic


**Paper:**
> Tang B, Wang X, Li Q, Bragazzi NL, Tang S, Xiao Y, Wu J. Estimation of the Transmission Risk of the 2019-nCoV and Its Implication for Public Health Interventions. J Clin Med. 2020 Feb 7;9(2):462. doi: 10.3390/jcm9020462. [PMID: 32046137](https://pubmed.ncbi.nlm.nih.gov/32046137/)

**Model:** [BIOMD0000000971]([text](https://www.ebi.ac.uk/biomodels/BIOMD0000000971))

<hr>

### General questions about the model


**Q1** Describe the model <br>

```
Briefly describe the BioModel 971
```


**Q2** Describe model components <br>

```
Describe model components
```

<hr>

### Compute infected cases over time
**Q3** Simulate the model and plot infected cases over time <br>
**Q4** Set the quarantine rate rate (`q_lockdown`) 20 times the initial value (=3.7774 e-06) and simulate the model (reproduce figure 3C from the paper). <br>
```
Simulate the model for 50 days with an interval of 50. Plot infected cases over time. Call this model_default.
```


**Q4** Set the quarantine rate (`q_lockdown`) 20 times the initial value (=3.7774 e-06) and simulate the model (reproduce the red line in figure 3C from the paper). <br>

```
Simulate the model again, but set the parameter 'q_lockdown' to 0.0000037774. Call this model_high_lockdown.
```

**Q5:** Compare infected cases for both simulations. <br>

```
What is the value of infected cases at the end of the simulation in both experiments?
```

<hr>



	
