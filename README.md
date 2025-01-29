[![Talk2BioModels](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2biomodels.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2biomodels.yml)
[![Talk2Cells](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2cells.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2cells.yml)
[![Talk2KnowledgeGraphs](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2knowledgegraphs.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2knowledgegraphs.yml)
[![Talk2Competitors](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2competitors.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2competitors.yml)
![GitHub Release](https://img.shields.io/github/v/release/VirtualPatientEngine/AIAgents4Pharma)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)


<h1 align="center" style="border-bottom: none;">🤖 AIAgents4Pharma</h1>

Welcome to **AIAgents4Pharma** – an open-source project by [Team VPE](https://github.com/VirtualPatientEngine) that brings together AI-driven tools to help researchers and pharma interact seamlessly with complex biological data.

Our toolkit currently consists of three intelligent agents, each designed to simplify and enhance access to specialized data in biology:

- **Talk2BioModels**: Engage directly with mathematical models in systems biology.
- **Talk2Cells** _(Work in progress)_: Query and analyze sequencing data with ease.
- **Talk2KnowledgeGraphs** _(Work in progress)_: Access and explore complex biological knowledge graphs for insightful data connections.
- **Talk2Competitors** _(Coming soon)_: Get recommendations for articles related to your choice. Download, query, and write/retrieve them to your reference manager (currently supporting Zotero).

---

## Overview of Agents

### 1. Talk2BioModels

**Talk2BioModels** is an AI agent designed to facilitate interaction with mathematical models in systems biology. Systems biology models are critical in understanding complex biological mechanisms, but they’re often inaccessible to those without coding or mathematical expertise. Talk2BioModels simplifies this, enabling researchers to focus on analysis and interpretation rather than on programming. With Talk2BioModels, users can interact directly with these models through natural language. By simply asking questions or making requests, users can:

- Forward simulation of both internal and open-source models (BioModels).
- Adjust parameters within the model to simulate different conditions.
- Query simulation results.
- Extract model information such as species, parameters, units and description.

### 2. Talk2Cells _(Work in Progress)_

**Talk2Cells** is being developed to provide direct access to and analysis of sequencing data, such as RNA-Seq or DNA-Seq, using natural language.

### 3. Talk2KnowledgeGraphs _(Work in Progress)_

**Talk2KnowledgeGraphs** is an agent designed to enable interaction with biological knowledge graphs (KGs). KGs integrate vast amounts of structured biological data into a format that highlights relationships between entities, such as proteins, genes, and diseases.

### 4. Talk2Competitors _(Coming soon)_

## Getting Started

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

### Installation

#### Option 1: PyPI

```bash
pip install aiagents4pharma
```

Check out the tutorials on each agent for detailed instrcutions.

#### Option 2: git

1. **Clone the repository:**

   ```bash
   git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma
   cd AIAgents4Pharma
   ```

2. **Install dependencies:**

   ```bash
   pip install .
   ```

3. **Initialize OPENAI_API_KEY**

   ```bash
   export OPENAI_API_KEY=....
   ```

4. **[Optional] Set up login credentials**

   ```bash
   vi .streamlit/secrets.toml
   ```

   and enter

   ```
   password='XXX'
   ```

   Please note that the passoword will be same for all the users.

5. **[Optional] Initialize LANGSMITH_API_KEY**

   ```bash
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY=<your-api-key>
   ```

   Please note that this will create a new tracing project in your Langsmith
   account with the name `<user_name>@<uuid>`, where `user_name` is the name
   you provided in the previous step. If you skip the previous step, it will
   default to `default`. <uuid> will be the 128 bit unique ID created for the
   session.

6. **Launch the app:**
   ```bash
   streamlit run app/frontend/streamlit_app.py
   ```

For detailed instructions on each agent, please refer to their respective folders.

---

## Usage

**Talk2BioModels** currently provides an interactive console where you can enter natural language queries to simulate models, adjust parameters, and query the simulated results.

More detailed usage examples, including sample data for Talk2Cells and Talk2KnowledgeGraphs, will be provided as development progresses.

---

## Contributing

We welcome contributions to AIAgents4Pharma! Here’s how you can help:

1. **Fork the repository**
2. **Create a new branch** for your feature (`git checkout -b feat/feature-name`)
3. **Commit your changes** (`git commit -m 'feat: Add new feature'`)
4. **Push to the branch** (`git push origin feat/feature-name`)
5. **Open a pull request**

### Current Needs

- **Beta testers** for Talk2BioModels.
- **Developers** with experience in natural language processing, bioinformatics, or knowledge graphs for contributions to AIAgents4Pharma.

Check out our [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

---

## Roadmap

### Completed

- **Talk2BioModels**: Initial release with core capabilities for interacting with systems biology models.

### Planned

- **User Interface**: Interactive web UI for all agents.
- **Talk2Cells**: Integration of sequencing data analysis tools.
- **Talk2KnowledgeGraphs**: Interface for biological knowledge graph interaction.
- **Talk2Competitors**: Interface for exploring articles

We’re excited to bring AIAgents4Pharma to the bioinformatics and pharmaceutical research community. Together, let’s make data-driven biological research more accessible and insightful.

**Get Started** with AIAgents4Pharma today and transform the way you interact with biological data.

---

## Feedback

Questions/Bug reports/Feature requests/Comments/Suggestions? We welcome all. Please use the `Isssues` tab 😀
