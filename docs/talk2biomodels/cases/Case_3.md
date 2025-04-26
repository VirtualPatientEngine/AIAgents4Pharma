# Example Use Case #3
## Exploring multiple steady states of MAPK phosphorylation cascade

<iframe width="672" height="378" src="https://www.youtube.com/embed/T5OjgjRSFno" frameborder="0" allowfullscreen></iframe>


**Paper:**
>Markevich NI, Hoek JB, Kholodenko BN. Signaling switches and bistability arising from multisite phosphorylation in protein kinase cascades. J Cell Biol. 2004 Feb 2;164(3):353-9. doi: 10.1083/jcb.200308060. Epub 2004 Jan 26. [PMID: 14744999](https://pubmed.ncbi.nlm.nih.gov/14744999/)

**Model** [BIOMD0000000027](https://www.ebi.ac.uk/biomodels/BIOMD0000000027)


<hr>
<i>Note: The specific responses to queries may differ from the ones provided here, depending on the release version of Talk2Biomodels and the large language model (LLM) utilized.</i>

<hr>

### General questions about the model

**Q:** Describe the model <br>

```
Briefly describe the BioModel 27
```

<details>
<summary>Answer</summary>

<h3>Model Title</h3>
<p>The BioModel 27 is titled "Markevich2004 - MAPK double phosphorylation, ordered Michaelis-Menton."</p>

<h3>Description</h3>
<p>This model corresponds to the schemas 1 and 2 of Markevich et al. (2004) and is modeled using Michaelis-Menten-like kinetics. It describes the phosphorylation and dephosphorylation processes that follow distributive ordered kinetics. The model successfully reproduces figure 3 from the main article.</p>

<h3>Citation</h3>
<p>The model is detailed in the article titled "Signaling switches and bistability arising from multisite phosphorylation in protein kinase cascades." by Markevich NI, Hoek JB, and Kholodenko BN, published in the Journal of Cell Biology in February 2004.</p>

<h3>Additional Information</h3>
<p>The model is hosted on the BioModels Database and can be identified by the identifier BIOMD0000000027.</p>
<p>It is dedicated to the public domain under the CC0 Public Domain Dedication.</p>

</details>



**Q:** Describe model components <br>

```
Describe model components
```

<details>
<summary>Answer</summary>


<h2>Model Components of BioModel 27</h2>
<h3>Species:</h3>
<ul>
  <li><strong>M</strong>
    <ul>
      <li>Compartment: uVol</li>
      <li>Type: Reactions</li>
      <li>Unit: nmol/l</li>
      <li>Initial Concentration: 500.0 nmol/l</li>
    </ul>
  </li>
  <li><strong>Mp</strong>
    <ul>
      <li>Compartment: uVol</li>
      <li>Type: Reactions</li>
      <li>Unit: nmol/l</li>
      <li>Initial Concentration: 0.0 nmol/l</li>
    </ul>
  </li>
  <li><strong>Mpp</strong>
    <ul>
      <li>Compartment: uVol</li>
      <li>Type: Reactions</li>
      <li>Unit: nmol/l</li>
      <li>Initial Concentration: 0.0 nmol/l</li>
    </ul>
  </li>
  <li><strong>MAPKK</strong>
    <ul>
      <li>Compartment: uVol</li>
      <li>Type: Fixed</li>
      <li>Unit: nmol/l</li>
      <li>Initial Concentration: 50.0 nmol/l</li>
    </ul>
  </li>
  <li><strong>MKP3</strong>
    <ul>
      <li>Compartment: uVol</li>
      <li>Type: Fixed</li>
      <li>Unit: nmol/l</li>
      <li>Initial Concentration: 100.0 nmol/l</li>
    </ul>
  </li>
</ul>

<h3>Parameters:</h3>
<ul>
  <li><strong>k1cat</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 0.01</li>
    </ul>
  </li>
  <li><strong>Km1</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 50.0</li>
    </ul>
  </li>
  <li><strong>k2cat</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 15.0</li>
    </ul>
  </li>
  <li><strong>Km2</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 500.0</li>
    </ul>
  </li>
  <li><strong>k3cat</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 0.084</li>
    </ul>
  </li>
  <li><strong>Km3</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 22.0</li>
    </ul>
  </li>
  <li><strong>k4cat</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 0.06</li>
    </ul>
  </li>
  <li><strong>Km4</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 18.0</li>
    </ul>
  </li>
  <li><strong>Km5</strong>
    <ul>
      <li>Type: Fixed</li>
      <li>Initial Value: 78.0</li>
    </ul>
  </li>
</ul>

<h3>Compartments:</h3>
<ul>
  <li>uVol</li>
</ul>

<h3>Units:</h3>
<ul>
  <li>Time Unit: seconds (s)</li>
  <li>Quantity Unit: nanomoles (nmol)</li>
  <li>Length Unit: meters (m)</li>
  <li>Area Unit: square meters (m²)</li>
  <li>Volume Unit: liters (l)</li>
</ul>

</details>

<hr>

### Model simulation

**Q:** What is the concentration of Mpp at a steady state? <br>

```
What is the concentration of Mpp at a steady state and what is its transition time until the steady state is reached?
```

<details>

<summary>Answer</summary>


The concentration of Mpp at steady state is <strong>49.4175 nmol/l</strong>, and the transition time until the steady state is reached is <strong>700 seconds</strong>.

<img src="../figures/C3_q3.png" width="600"/>

</details>



<hr>

### Emergent model behavior (hysteresis)

**Q:** Observe the steady state concentration of Mpp if the Mpp is in "ON" state.<br>

```
Set initial concentration of MAPKK to 50, Mpp to 350 and M to 150, and then bring the model to a steady state. Call this experiment_on.
```

```

What is the steady state concentration of Mpp in this experiment?
```

<details>
<summary>Answer</summary>


The steady state concentration of Mpp in the experiment "experiment_on" is <strong>277.26 nmol/l</strong>.

</details>


**Q:** Observe the steady state concentration of Mpp if the Mpp is in "OFF" state.<br>

```
Set initial concentration of MAPKK to 50, Mpp to 150 and M to 350, and then bring the model to a steady state. Call this experiment_off.
```



```
What is the steady state concentration of Mpp in this experiment?
```
<details>
<summary>Answer</summary>

The steady state concentration of Mpp in the experiment "experiment_off" is <strong>49.4175 nmol/l</strong>.

</details>

**Q:** Evaluate the effect of MAPKK concentration on the steady state concentration of Mpp<br>

```
How will the concentration of Mpp change, if the initial value of MAPKK were to be changed between 1 and 100 in steps of 10?
```

<details>

<summary>Answer</summary>
<img src="../figures/C3_q6.png" width="600"/>
</details>
