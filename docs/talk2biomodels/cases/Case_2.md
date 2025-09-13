# Example Use Case #2
## Predicting effectiveness of contact inhibition during SARS-CoV-2 virus pandemic

<iframe width="672" height="378" src="https://www.youtube.com/embed/6qnMPBrKfQg" frameborder="0" allowfullscreen></iframe>

**Paper:**
> Tang B, Wang X, Li Q, Bragazzi NL, Tang S, Xiao Y, Wu J. Estimation of the Transmission Risk of the 2019-nCoV and Its Implication for Public Health Interventions. J Clin Med. 2020 Feb 7;9(2):462. doi: 10.3390/jcm9020462. [PMID: 32046137](https://pubmed.ncbi.nlm.nih.gov/32046137/)

**Model:** [BIOMD0000000971](https://www.ebi.ac.uk/biomodels/BIOMD0000000971)

<hr>

<i>Note: The specific responses to queries may differ from the ones provided here, depending on the release version of Talk2Biomodels and the large language model (LLM) utilized.</i>

<hr>

### General questions about the model


**Q:** Describe the model <br>

```

Briefly describe the BioModel 971
```
<details>
<summary>Answer</summary>

The BioModel 971, named "Tang2020 - Estimation of transmission risk of COVID-19 and impact of public health interventions," focuses on the rapid spread of the novel coronavirus (2019-nCoV) since its emergence in Wuhan, China. It utilizes a deterministic compartmental model to estimate the basic reproduction number, which is crucial for understanding the potential severity of the outbreak and guiding disease intervention strategies.

The model indicates that the control reproduction number could be as high as 6.47, and sensitivity analyses suggest that interventions like intensive contact tracing, quarantine, and isolation can significantly reduce transmission risk. The study highlights the effectiveness of travel restrictions implemented by Chinese authorities, showing that such measures can lead to a substantial decrease in the number of infected individuals. Under strict measures, the outbreak is expected to peak within two weeks, with a notable reduction in cases due to travel restrictions.

</details>

**Q:** Describe model components <br>

```
Describe model components
```

<details>

<summary>Answer</summary>

The components of BioModel 971 include:
<h3>Species:</h3>
<ul>
  <li><strong>Susceptible:</strong> Individuals who are at risk of contracting the virus.</li>
  <li><strong>Exposed:</strong> Individuals who have been exposed to the virus but are not yet infectious.</li>
  <li><strong>Infected:</strong> Individuals who are currently infected and can transmit the virus.</li>
  <li><strong>Asymptomatic:</strong> Infected individuals who do not show symptoms but can still spread the virus.</li>
  <li><strong>Susceptible_quarantined:</strong> Susceptible individuals who are in quarantine.</li>
  <li><strong>Exposed_quarantined:</strong> Exposed individuals who are in quarantine.</li>
  <li><strong>Hospitalised:</strong> Individuals who are hospitalized due to severe infection.</li>
  <li><strong>Recovered:</strong> Individuals who have recovered from the infection.</li>
</ul>

<h3>Parameters:</h3>
<ul>
  <li><strong>c:</strong> Contact rate.</li>
  <li><strong>beta:</strong> Transmission probability per contact.</li>
  <li><strong>q:</strong> Quarantine rate.</li>
  <li><strong>sigma:</strong> Rate of progression from exposed to infected.</li>
  <li><strong>lambda:</strong> Rate of recovery.</li>
  <li><strong>rho:</strong> Rate of hospitalization.</li>
  <li><strong>delta_I:</strong> Rate of death for infected individuals.</li>
  <li><strong>delta_q:</strong> Rate of death for quarantined individuals.</li>
  <li><strong>gamma_I:</strong> Recovery rate for infected individuals.</li>
  <li><strong>gamma_A:</strong> Recovery rate for asymptomatic individuals.</li>
  <li><strong>gamma_H:</strong> Recovery rate for hospitalized individuals.</li>
  <li><strong>alpha:</strong> Rate of symptomatic infection.</li>
  <li><strong>theta:</strong> Effectiveness of interventions.</li>
  <li><strong>c_lockdown:</strong> Contact rate during lockdown.</li>
  <li><strong>q_lockdown:</strong> Quarantine rate during lockdown.</li>
  <li><strong>Infected:</strong> Number of infected individuals.</li>
</ul>

<h3>Compartments:</h3>
<ul>
  <li><strong>Wuhan:</strong> The geographical compartment where the model is focused.</li>
</ul>

<h3>Units:</h3>
<ul>
  <li><strong>Time Unit:</strong> Days (d)</li>
  <li><strong>Quantity Unit:</strong> Number of individuals (#)</li>
  <li><strong>Length Unit:</strong> Meters (m)</li>
  <li><strong>Area Unit:</strong> Square meters (m²)</li>
  <li><strong>Volume Unit:</strong> Milliliters (ml)</li>
</ul>

<p>These components work together to simulate the dynamics of the COVID-19 outbreak and the impact of public health interventions.</p>
</details>


<hr>

### Compute infected cases over time
**Q:** Simulate the model and plot infected cases over time <br>

```
Simulate the model for 50 days with an interval of 50. Plot infected cases over time. Call this model_default.
```

<details>

<summary>Answer</summary>
<img src="../figures/C2_q3.png" width="600"/>
</details>


**Q:** Set the quarantine rate (`q_lockdown`) 20 times the initial value (=3.7774 e-06) and simulate the model (reproduce the red line in figure 3C from the paper). <br>



```
Simulate the model again, but set the parameter 'q_lockdown' to 0.0000037774. Call this model_high_lockdown.
```

```
Plot the infected cases over time.
```

<details>
<summary>Answer</summary>
<img src="../figures/C2_q4.png" width="600"/>
</details>


**Q:** Compare infected cases for both simulations. <br>

```
What is the value of infected cases at the end of the simulation in both experiments?
```

<details>
<summary>Answer</summary>

The values of infected cases at the end of the simulations are as follows:
<ul>
  <li>For the experiment model_default, the value is approximately <strong>104,338.52</strong></li>
  <li>For the experiment model_high_lockdown, the value is approximately <strong>7,275.49</strong></li>
</ul>
</details>
