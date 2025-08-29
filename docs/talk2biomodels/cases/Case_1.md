# Example Use Case #1
## Crohn's Disease model

<iframe width="672" height="378" src="https://www.youtube.com/embed/Mp9quRxtE10" frameborder="0" allowfullscreen></iframe>

**Paper:**
> Dwivedi G, Fitz L, Hegen M, Martin SW, Harrold J, Heatherington A, Li C. (2014). A multiscale model of interleukin-6-mediated immune regulation in Crohn's disease and its application in drug discovery and development. *CPT Pharmacometrics Syst Pharmacol*, 3(1):e89. doi: 10.1038/psp.2013.64. [PMID: 24402116](https://pubmed.ncbi.nlm.nih.gov/24402116/)

**Model**: [BIOMD0000000537](https://www.ebi.ac.uk/biomodels/BIOMD0000000537)

<hr>

<i>Note: The specific responses to queries may differ from the ones provided here, depending on the release version of Talk2Biomodels and the large language model (LLM) utilized.</i>

<hr>

### General questions about the model

<hr>

**Q:** Search for a model on BioModels database related to a search term <br>

```
Search for all the BioModels on Crohn's Disease
```

<details>
<summary>Answer</summary>

<table>
<thead>
<tr>
<th>#</th>
<th>BioModel ID</th>
<th>BioModel Name</th>
<th>Format</th>
<th>Submission Date</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>BIOMD0000000535</td>
<td>Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6 Antibody</td>
<td>SBML</td>
<td>2014-08-04</td>
</tr>
<tr>
<td>2</td>
<td>BIOMD0000000536</td>
<td>Dwivedi2014 - Crohns IL6 Disease model - sgp130 activity</td>
<td>SBML</td>
<td>2014-08-04</td>
</tr>
<tr>
<td>3</td>
<td>BIOMD0000000537</td>
<td>Dwivedi2014 - Crohns IL6 Disease model - Anti-IL6R Antibody</td>
<td>SBML</td>
<td>2014-08-04</td>
</tr>
</tbody>
</table>
</details>


**Q:** Get model description <br>

```
Briefly describe the BioModel 537
```

<details>
<summary>Answer</summary>

<h3>Description:</h3>
<p>This model consists of four interconnected models that explore different aspects of Crohn's disease related to Interleukin-6 (IL-6) inhibition. The models include:</p>

<ul>
<li>Healthy Volunteer model</li>
<li>Crohn's Disease - IL-6 Antibody</li>
<li>Crohn's Disease - sgp130FC</li>
<li>Crohn's Disease - IL-6Ra Antibody</li>
</ul>

<p>The primary focus is on simulating the differential activity of various ligands on IL-6 signaling, which impacts the activity of Signal Transducer and Activator of Transcription 3 (STAT3) and the production of the biomarker C-Reactive Protein (CRP).</p>

<p>The model is detailed in the article titled <i>A multiscale model of interleukin-6-mediated immune regulation in Crohn's disease and its application in drug discovery and development</i>, published in CPT Pharmacometrics & Systems Pharmacology in 2014.</p>

<p>This model is hosted on the BioModels Database and can be identified by the ID BIOMD0000000537.</p>

</details>

**Q:** Ask about publication data of the model (It will remember the model number) <br>

```
When and where was this model published?
```

<details>
<summary>Answer</summary>

<h3>Publication Information</h3>
<p>The model was published in the article titled "A multiscale model of interleukin-6-mediated immune regulation in Crohn's disease and its application in drug discovery and development."</p>

<h4>Publication Details:</h4>
<ul>
  <li><strong>Authors:</strong> Dwivedi G, Fitz L, Hegen M, Martin SW, Harrold J, Heatherington A, Li C.</li>
  <li><strong>Journal:</strong> CPT Pharmacometrics & Systems Pharmacology</li>
  <li><strong>Publication Date:</strong> 2014</li>
  <li><strong>Volume:</strong> 3</li>
  <li><strong>Article Identifier:</strong> e89</li>
  <li><strong>Link to Article:</strong> PubMed</li>
</ul>

<p>This study discusses the development of a multiscale systems model for understanding IL-6-mediated immune regulation in Crohn's disease and its implications for drug discovery.</p>
</details>

<hr>

### Interrogate model species and annotations


**Q:** Display all parameters related to drug dosage. <br>

```
Show me the parameters related to drug dosing regimen in the model
```

<details>
<summary>Answer</summary>

<p>The parameters related to the drug dosing regimen in the BioModel 537 include:</p>

<ul>
  <li><strong>Dose:</strong> The amount of drug administered in a single dose.</li>
  <li><strong>DoseQ2W:</strong> The dosage administered every two weeks.</li>
  <li><strong>Initial for Dose:</strong> Initial concentration for the standard dose.</li>
  <li><strong>Initial for DoseQ2W:</strong> Initial concentration for the dosage administered every two weeks.</li>
  <li><strong>infusionTime:</strong> The duration of the drug infusion.</li>
</ul>

<p>These parameters are crucial for defining how the drug is administered over time and help in modeling the effects of different dosing schedules on the pharmacokinetics and pharmacodynamics of the drug.</p>

</details>
<hr>

**Q:** Show annotated species of the model _(This step might take time.)_. <br>

```
show me the annotations of all species in the model 537
```

<details>
<summary>Answer (snippet)</summary>
<img src="../figures/C1_q4.2.png" width="600"/>
</details>

**Q:** Show all species of a specific type in the model <br>

```
Show annotations of all interleukins in model 537
```

<details>
<summary>Answer</summary>
<img src="../figures/C1_q4.3.png" width="600"/>
</details>

**Q:** Interrogate specific species in the model <br>

```
what is R in this model?
```

<details>
<summary>Answer</summary>
<img src="../figures/C1_q4.4.png" width="600"/>
</details>

### Ask questions to the published article using Retrieval-Augmented Generation (RAG)

 **Q:** Ask questions to the published article concerning the model species. To utilize Retrieval-Augmented Generation (RAG) for answering questions about the model, the PDF file of the article should be uploaded first.
 The article can be found [here](https://ascpt.onlinelibrary.wiley.com/doi/epdf/10.1038/psp.2013.64).


```
Query the uploaded article to suggest a possible description and role of the parameter R in the model.
```

<details>
<summary>Answer</summary>

<p>The parameter R in the model is associated with the dynamics of IL-6 signaling in the context of Crohn's disease. While the specific role of R is not explicitly detailed in the provided excerpts, it can be inferred that R is likely involved in the regulation of receptor interactions or signaling pathways related to IL-6.</p>

<h4>Possible Description and Role of Parameter R:</h4>
<p><strong>Description:</strong> R may represent a receptor or a regulatory component in the IL-6 signaling pathway, potentially influencing the binding and activity of IL-6 or its receptors (such as IL-6Rα).</p>
<p><strong>Role:</strong> The parameter R could play a critical role in modulating the effects of IL-6 signaling, particularly in how the immune system responds to IL-6 in the context of Crohn's disease. It may affect the concentration of downstream signaling molecules or biomarkers, such as C-Reactive Protein (CRP), which is a key indicator of inflammation in this disease.</p>

<p>In summary, while the exact function of R is not explicitly stated, it is likely integral to the signaling dynamics of IL-6 and its impact on immune regulation in Crohn's disease. Further details from the model or additional context from the article may provide more specific insights into its role.</p>


</details>

**Q:** Access simulation time from the publication. To utilize Retrieval-Augmented Generation (RAG) for answering questions about the model, the PDF file of the article should be uploaded first. The article can be found [here](https://ascpt.onlinelibrary.wiley.com/doi/epdf/10.1038/psp.2013.64).

```
Query the uploaded article and return the model's simulation duration time mentioned in the paper.
```

<details>
<summary>Answer</summary>

<p>The model in the paper simulates the pharmacological effects over a duration of <strong>12 weeks</strong>. This is indicated in the context of intravenous drug administration at 4-week intervals, where the effects on biomarkers such as serum CRP are assessed throughout this period.</p>

</details>



**Q:** Show concentrations of model species <br>

```
show concentrations of all species in this model in a table format

```

<details>
<summary>Answer</summary>

<table>
<tr><th>Species Name</th><th>Compartment</th><th>Initial Concentration (nmol/l)</th></tr>
<tr><td>IL6{serum}</td><td>Serum</td><td>0.0004356</td></tr>
<tr><td>sgp130{serum}</td><td>Serum</td><td>3.9</td></tr>
<tr><td>sR_IL6_sgp130{serum}</td><td>Serum</td><td>0.0874</td></tr>
<tr><td>CRP{serum}</td><td>Serum</td><td>221.064</td></tr>
<tr><td>sR{serum}</td><td>Serum</td><td>4.2535</td></tr>
<tr><td>sR_IL6{serum}</td><td>Serum</td><td>0.0011</td></tr>
<tr><td>Ab{serum}</td><td>Serum</td><td>2.3818e-29</td></tr>
<tr><td>Ab_sR{serum}</td><td>Serum</td><td>6.1044e-26</td></tr>
<tr><td>Ab_sR_IL6{serum}</td><td>Serum</td><td>-7.4133e-29</td></tr>
<tr><td>CRP Suppression (%)</td><td>Serum</td><td>-0.0</td></tr>
<tr><td>CRP (% of baseline)</td><td>Serum</td><td>100.0</td></tr>
<tr><td>gp130{liver}</td><td>Liver</td><td>0.3889</td></tr>
<tr><td>R_IL6_gp130{liver}</td><td>Liver</td><td>6.5994e-05</td></tr>
<tr><td>sR_IL6{liver}</td><td>Liver</td><td>0.0009762</td></tr>
<tr><td>R</td><td>Liver</td><td>0.4382</td></tr>
<tr><td>IL6{liver}</td><td>Liver</td><td>0.0007257</td></tr>
<tr><td>R_IL6</td><td>Liver</td><td>1.6004e-05</td></tr>
<tr><td>Ractive{liver}</td><td>Liver</td><td>0.7654</td></tr>
<tr><td>STAT3{liver}</td><td>Liver</td><td>0.7775</td></tr>
<tr><td>pSTAT3{liver}</td><td>Liver</td><td>9.2225</td></tr>
<tr><td>CRP{liver}</td><td>Liver</td><td>158.325</td></tr>
<tr><td>sR{liver}</td><td>Liver</td><td>6.0870</td></tr>
<tr><td>CRPExtracellular</td><td>-</td><td>409.775</td></tr>
<tr><td>sgp130{liver}</td><td>Liver</td><td>5.5897</td></tr>
<tr><td>sR_IL6_sgp130{liver}</td><td>Liver</td><td>0.1163</td></tr>
<tr><td>Ab_sR{liver}</td><td>Liver</td><td>8.6162e-26</td></tr>
<tr><td>Ab{liver}</td><td>Liver</td><td>2.2002e-29</td></tr>
<tr><td>Ab_R</td><td>-</td><td>2.0337e-27</td></tr>
<tr><td>Ab_sR_IL6{liver}</td><td>Liver</td><td>-3.0779e-29</td></tr>
<tr><td>sR_IL6{gut}</td><td>Gut</td><td>0.0013</td></tr>
<tr><td>gp130{gut}</td><td>Gut</td><td>0.3750</td></tr>
<tr><td>R_IL6_gp130{gut}</td><td>Gut</td><td>8.4489e-05</td></tr>
<tr><td>Ractive{gut}</td><td>Gut</td><td>0.9803</td></tr>
<tr><td>STAT3{gut}</td><td>Gut</td><td>0.6106</td></tr>
<tr><td>pSTAT3{gut}</td><td>Gut</td><td>9.3894</td></tr>
<tr><td>geneProduct</td><td>Gut</td><td>159.803</td></tr>
<tr><td>sR{gut}</td><td>Gut</td><td>6.0641</td></tr>
<tr><td>IL6{gut}</td><td>Gut</td><td>0.0093</td></tr>
<tr><td>sgp130{gut}</td><td>Gut</td><td>5.5697</td></tr>
<tr><td>sR_IL6_sgp130{gut}</td><td>Gut</td><td>0.1363</td></tr>
<tr><td>Ab{gut}</td><td>Gut</td><td>2.1588e-29</td></tr>
<tr><td>Ab_sR{gut}</td><td>Gut</td><td>2.6206e-26</td></tr>
<tr><td>Ab_sR_IL6{gut}</td><td>Gut</td><td>-1.2092e-28</td></tr>
<tr><td>Ab{peripheral}</td><td>Peripheral</td><td>1.6792e-29</td></tr>
</table>

</details>

**Q:** Show parameter values <br>

```
show parameter values in this model
```

<details>
<summary>Answer</summary>
<table>
<tr><th>Parameter Name</th><th>Initial Value</th><th>Unit</th></tr>
<tr><td>Values[kRLOn]</td><td>0.384</td><td></td></tr>
<tr><td>Values[kRLOff]</td><td>1.92</td><td></td></tr>
<tr><td>Values[kgp130On]</td><td>20.52</td><td></td></tr>
<tr><td>Values[kgp130Off]</td><td>1.026</td><td></td></tr>
<tr><td>Values[kRAct]</td><td>155.0</td><td></td></tr>
<tr><td>Values[kRint]</td><td>1.96</td><td></td></tr>
<tr><td>Values[kRsynth]</td><td>0.0685</td><td></td></tr>
<tr><td>Values[kRintBasal]</td><td>0.1561</td><td></td></tr>
<tr><td>Values[ksynthIL6]</td><td>0.0063</td><td></td></tr>
<tr><td>Values[kdegIL6]</td><td>34.82</td><td></td></tr>
<tr><td>Values[kdegCRP]</td><td>0.36</td><td></td></tr>
<tr><td>Values[KmSTATDephos]</td><td>5.34</td><td></td></tr>
<tr><td>Values[VmSTATDephos]</td><td>0.62</td><td></td></tr>
<tr><td>Values[VmRDephos]</td><td>0.525</td><td></td></tr>
<tr><td>Values[KmRDephos]</td><td>155.3</td><td></td></tr>
<tr><td>Values[kcatSTATPhos]</td><td>145.0</td><td></td></tr>
<tr><td>Values[KmSTATPhos]</td><td>219.0</td><td></td></tr>
<tr><td>Values[KmProtSynth]</td><td>10.0</td><td></td></tr>
<tr><td>Values[VmProtSynth]</td><td>330.0</td><td></td></tr>
<tr><td>Values[kCRPSecretion]</td><td>0.5</td><td></td></tr>
<tr><td>Values[ksynthCRP]</td><td>0.42</td><td></td></tr>
<tr><td>Values[ksynthsR]</td><td>0.1</td><td></td></tr>
<tr><td>Values[kdegsR]</td><td>0.3</td><td></td></tr>
<tr><td>Values[ksynthsgp130]</td><td>3.9</td><td></td></tr>
<tr><td>Values[kdegsgp130]</td><td>1.0</td><td></td></tr>
<tr><td>Values[ksynthIL6Gut]</td><td>0.036</td><td></td></tr>
<tr><td>Values[kdegIL6Gut]</td><td>1.0</td><td></td></tr>
<tr><td>Values[kdistTissueToSerum]</td><td>0.8473</td><td></td></tr>
<tr><td>Values[kdistSerumToTissue]</td><td>1.2125</td><td></td></tr>
<tr><td>Values[kRShedding]</td><td>0.0054</td><td></td></tr>
<tr><td>Values[kintActiveR]</td><td>0.01</td><td></td></tr>
<tr><td>Values[kIL6RBind]</td><td>1000.0</td><td></td></tr>
<tr><td>Values[kIL6RUnbind]</td><td>2.5</td><td></td></tr>
<tr><td>Values[infusionTime]</td><td>1.0</td><td></td></tr>
<tr><td>Values[kAbSerumToLiver]</td><td>0.0208</td><td></td></tr>
<tr><td>Values[kAbLiverToSerum]</td><td>0.0208</td><td></td></tr>
<tr><td>Values[kAbSerumToGut]</td><td>0.0104</td><td></td></tr>
<tr><td>Values[kAbGutToSerum]</td><td>0.0208</td><td></td></tr>
<tr><td>Values[VSerum]</td><td>2.88</td><td></td></tr>
<tr><td>Values[VLiver]</td><td>2.88</td><td></td></tr>
<tr><td>Values[VGut]</td><td>1.44</td><td></td></tr>
<tr><td>Values[VPeriph]</td><td>0.576</td><td></td></tr>
<tr><td>Values[QSerumLiver]</td><td>0.06</td><td></td></tr>
<tr><td>Values[QSerumGut]</td><td>0.03</td><td></td></tr>
<tr><td>Values[QSerumPeriph]</td><td>0.001</td><td></td></tr>
<tr><td>Values[kAbSerumToPeriph]</td><td>0.000347</td><td></td></tr>
<tr><td>Values[kAbPeriphToSerum]</td><td>0.001736</td><td></td></tr>
<tr><td>Values[kdegAb]</td><td>0.0018</td><td></td></tr>
<tr><td>Values[Dose]</td><td>300.0</td><td></td></tr>
<tr><td>Values[DoseQ2W]</td><td>0.0</td><td></td></tr>
<tr><td>Values[Initial for CRP]</td><td>221.0637</td><td></td></tr>
<tr><td>Values[Initial for DoseQ2W]</td><td>0.0</td><td></td></tr>
<tr><td>Values[Initial for Dose]</td><td>300.0</td><td></td></tr>
</table>

</details>

<hr>



### Model simulation and plotting

**Q:** Simulate the model for 12 weeks (= 2016 hours), and give the simulation result a name. <br>

```
Simulate the model for 2016 hours with 300 intervals. Set the initial concentration of Dose to 200 mg. Call this result `Treatment 4wk`.
```

<details>
<summary>Answer</summary>

<h3>Figure:</h3>
<img src="../figures/C1_q5.png" width="600"/>


<h3>Table:</h3>
<table>
  <tr>
    <th>Time</th>
    <th>sR{serum}</th>
    <th>sgp130{serum}</th>
    <th>R_IL6_gp130{liver}</th>
    <th>IL6{serum}</th>
    <th>Ab{serum}</th>
    <th>R</th>
    <th>...</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>4.253507</td>
    <td>3.900000</td>
    <td>6.599359e-05</td>
    <td>0.000436</td>
    <td>2.381820e-29</td>
    <td>0.438236</td>
    <td>...</td>
  </tr>
  <tr>
    <td>1.0</td>
    <td>0.000031</td>
    <td>3.901765</td>
    <td>6.420660e-05</td>
    <td>0.000638</td>
    <td>6.753452e+02</td>
    <td>0.000178</td>
    <td>...</td>
  </tr>
  <tr>
    <td>2.0</td>
    <td>0.000037</td>
    <td>3.905215</td>
    <td>6.248857e-05</td>
    <td>0.000739</td>
    <td>6.522828e+02</td>
    <td>0.000070</td>
    <td>...</td>
  </tr>
  <tr>
    <td>3.0</td>
    <td>0.000043</td>
    <td>3.907882</td>
    <td>6.013415e-05</td>
    <td>0.000756</td>
    <td>6.303828e+02</td>
    <td>0.000049</td>
    <td>...</td>
  </tr>
  <tr>
    <td>4.0</td>
    <td>0.000049</td>
    <td>3.909825</td>
    <td>5.762284e-05</td>
    <td>0.000755</td>
    <td>6.095828e+02</td>
    <td>0.000039</td>
    <td>...</td>
  </tr>
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
</table>
</details>

**Q:** Show only CRP-related species, as it is one of the clinically-relevant measurement for a successful treatment. <br>

```
Plot only CRP-related species.
```

<details>
<summary>Answer</summary>

<h3>Figure:</h3>
<img src="../figures/C1_q6.png" width="600"/>
</details>

**Q:** Review, how drug treatment influences CRP concentration in the serum at any arbitrary time point, e.g. 12 weeks after the initial drug treatment (2016 hours). <br>

```
What is the CRP concentration in the serum at the end of the simulation?
```

<details>
<summary>Answer</summary>

<p>The CRP concentration in the serum at the end of the simulation is approximately <strong>220.52 nmol</strong>.</p>

</details>

<hr>

### Compare two treatment regimens and apply reasoning


**Q:** Administrate the drug treatment every two weeks by activating the `DoseQ2W` and deactivating `Dose` parameters. <br>


```
Simulate the model again for 2016 hours with 300 intervals. Set the initial concentration of `DoseQ2W` to 200 and `Dose` to 0. Call this result `Treatment 2wk`.
```
<details>
<summary>Answer</summary>
<img src="../figures/C1_q8.png" width="600"/>

</details>

**Q:** Compare the CRP values in both aforementioned cases. 'Treatment 2wk' should reduce CRP earlier than 'Treatment 4wk'. <br>

```
Based on the CRP values in serum at the end of the simulation, which treatment would you recommend `Treatment 4wk` or `Treatment 2wk`?
```

<details>
<summary>Answer</summary>

<p>At the end of the simulations, the CRP concentrations in the serum are as follows:</p>

<ul>
  <li><strong>Treatment 4wk:</strong> Approximately 220.52 nmol</li>
  <li><strong>Treatment 2wk:</strong> Approximately 1.457 nmol</li>
</ul>

<p>Based on these values, I would recommend <strong>Treatment 2wk</strong>, as it results in a significantly lower CRP concentration in the serum compared to Treatment 4wk. A lower CRP concentration indicates better suppression of inflammation. If you have any further questions or need additional insights, feel free to ask!</p>

</details>

<hr>

### Compute the concentration of free drug in blood serum


**Q:** Reproduce Figure 4f from the paper for a 500mg dose. <br>

```
Set the initial concentration of 'Dose' to 500 mg and simulate the model for 2016 hours with 300 intervals.
```
```
Plot Ab in serum.
```


<details>
<summary>Answer</summary>

<h3>Figure:</h3>
<img src="../figures/C1_q10.png" width="600"/>

</details>

<hr>

### Simulate two antibodies with varying dissociation affinities

**Q:** Plot the current trajectory of CRP % suppression with the initial model parameters (`kIL6RUnbind` = 2.5). Compare the CRP suppression after 12 weeks of treatment. The authors expect 100% suppression of CRP after 12 weeks. <br>

```
Simulate the model 537 for 2016 hours with 300 intervals. Save this simulation as 'Ab_2.5'.
```
```
Plot the trajectory of CRP % suppression.
```


<details>
<summary>Answer</summary>

<img src="../figures/C1_q11.png" width="600"/>

</details>


**Q:** Set the parameter `kIL6RUnbind` to 250, decreasing the stability of the antibody-IL6R complex. Simulate the model and plot serum CRP % suppression. The authors expect ~10% suppression of CRP after 12 weeks. <br>

```
Simulate the model for 2016 hours with 300 intervals, but set the parameter `kIL6RUnbind` to 250. Save this simulation as 'Ab_250'.
```
```
Plot the curve of CRP % suppression.
```

<details>
<summary>Answer</summary>

<img src="../figures/C1_q12.png" width="600"/>

</details>



**Q:** Inquire about the effectiveness of the antibodies. <br>

```
Which antibody 'Ab_250' or 'Ab_2.5' is more effective in reducing CRP in the blood serum at the end of the simulation?
```

<details>
<summary>Answer</summary>
<p>At the end of the simulation:</p>
<ul>
  <li>Ab_250 resulted in a final CRP level of approximately 219.77 nmol.</li>
  <li>Ab_2.5 resulted in a final CRP level of approximately 2.27 nmol.</li>
</ul>

<p>Based on this data, Ab_2.5 is more effective in reducing CRP in the blood serum at the end of the simulation. If you have any further questions or need additional analysis, feel free to ask!</p>
</details>

<hr>

### Parameter scanning
**Q:** Analyze the variability of the parameter kIL6RUnbind to evaluate how CRP levels in serum are influenced by the stability of the antibody-IL6R complex. <br>

```
How will the concentration of CRP in serum change, if the parameter `kIL6RUnbind` were to be changed from 2 to 250 in steps of 20?
```

<details>
<summary>Answer</summary>
<img src="../figures/C1_q14.png" width="600"/>
</details>
