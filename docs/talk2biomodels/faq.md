### T2B - Frequently Asked Questions (FAQs)

#### 1. How can I find models related to a specific disease or topic?
<details>
  <summary>Click to expand</summary>
  You can use the `Search_models` tool to query the BioModels database for relevant models based on keywords. Here are some examples:
  <br>
  - Search for Crohn’s disease models:
    ```
    Search models on Crohn’s disease
    ```
  - Find models on COVID-19:
    ```
    Find models on COVID-19
    ```
</details>

#### 2. How can I get details about a specific model?
<details>
  <summary>Click to expand</summary>
  Use the `Get_modelinfo` tool to retrieve information such as species names, parameters, initial values, and descriptions. Examples:
  <br>
  - Briefly describe a model:
    ```
    Briefly describe model 971
    ```
  - List all species and parameters in a model:
    ```
    List all the species and parameters in model 971
    ```
  - Get the initial concentration of a specific species:
    ```
    What is the initial concentration of Pyruvate in model 64?
    ```
  - List model name and units in a table format:
    ```
    List the name and units in model 537. Show them as a table.
    ```
</details>

#### 3. How do I retrieve annotations from a model?
<details>
  <summary>Click to expand</summary>
  The `Get_annotations` tool extracts model annotations and provides descriptions from UniProt and OLS databases where possible.
  <br>
  - Show all annotations in a model:
    ```
    Show me all the annotations in model 64
    ```
  - Retrieve specific annotations related to interleukins:
    ```
    Show me annotations of only interleukin-related species in model 537
    ```
</details>

#### 4. How do I run a simulation on a model?
<details>
  <summary>Click to expand</summary>
  Use the `Simulate_model` tool to simulate a model. You can specify duration, time intervals, and initial values for species/parameters.
  <br>
  - Basic simulation:
    ```
    Simulate model 64
    ```
  - Specify simulation time and interval:
    ```
    Simulate model 64 for 3 days with an interval of 10
    ```
  - Set initial concentration and add recurring events:
    ```
    Run a simulation on model 537 for 100 hours with a time interval of 50.
    Set the initial concentration of Ab{serum} to 100.
    Add a recurring event that resets Ab{serum} to 100 every 20 hours.
    ```
</details>

#### 5. Can I create custom plots from my simulation results?
<details>
  <summary>Click to expand</summary>
  The `Custom_plotter` tool helps you focus on a subset of species in a simulation plot.
  <br>
  - Plot only CRP-related species:
    ```
    Make a custom plot to show only CRP-related species.
    ```
  - Plot glucose-related species:
    ```
    Plot only glucose-related species.
    ```
</details>

#### 6. How do I bring a model to steady state?
<details>
  <summary>Click to expand</summary>
  The `Steady_state` tool helps stabilize a model.
  <br>
  - Basic steady-state analysis:
    ```
    Bring model 64 to a steady state
    ```
  - Set initial conditions before analysis:
    ```
    Bring model 64 to a steady state. Set the initial concentration of Pyruvate to 50.
    ```
</details>

#### 7. How do I ask questions about the simulation results?
<details>
  <summary>Click to expand</summary>
  Use the `Ask_question` tool after running a simulation or steady-state analysis to retrieve insights.
  <br>
  - Steady-state concentration of a species:
    ```
    What is the steady-state concentration of Pyruvate?
    ```
  - Time required for a species to reach steady state:
    ```
    How long does glucose-related species take to reach steady state?
    ```
  - Final concentration in a simulation:
    ```
    What is the concentration of CRP-related species at the end of the simulation?
    ```
</details>

#### 8. How do I analyze how one species or parameter affects another?
<details>
  <summary>Click to expand</summary>
  The `Parameter_scan` tool allows you to analyze how changes in one parameter affect another.
  <br>
  - Effect of extracellular glucose on Pyruvate:
    ```
    How does the value of Pyruvate change in model 64 if the concentration of Extracellular Glucose is changed from 10 to 100 with a step size of 10? The simulation should run for 5 time units with an interval of 10.
    ```
  - Effect of dose changes on CRP concentration:
    ```
    Run a param scan in model 537 to observe change in concentration of CRP in serum over time if initial value of the parameter Dose is changed from 100 to 500 with a step size of 25.
    ```
</details>

#### 9. Can I ask questions about a research article I uploaded?
<details>
  <summary>Click to expand</summary>
  The `Query_article` tool helps analyze uploaded research articles.
  <br>
  - Summarize key takeaways:
    ```
    What are the key takeaways of the uploaded article?
    ```
  - Find parameter descriptions in the article:
    ```
    Query the uploaded article to suggest possible description of the parameter gR in model 64.
    ```
</details>

#### 10. How can I compare different simulations or steady-state results?
<details>
  <summary>Click to expand</summary>
  Assign names to different experiments and compare results.
  <br>
  - Compare simulations:
    ```
    Simulate model 537 for 2016 hours. Call it experiment Treatment4wk.
    ```
    ```
    Redo the simulation for 2016 hours but set the initial value of Dose to 0 and DoseQ2W to 300. Call it experiment Treatment2wk.
    ```
    ```
    What is the concentration of CRP in serum at the end of the simulation in all experiments done so far?
    ```
  - Compare steady-state results:
    ```
    Bring model 64 to a steady state. Call it Experiment_1.
    ```
    ```
    Bring model 64 to a steady state. Set initial concentration of NADH to 10. Call it Experiment_2.
    ```
    ```
    Compare the concentration of Pyruvate in both experiments at steady state.
    ```
</details>
