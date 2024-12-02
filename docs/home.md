<h1 align="center" style="border-bottom: none;">🤖 AIAgents4Pharma</h1>

Welcome to **AIAgents4Pharma** – an open-source project by [Team VPE](https://github.com/VirtualPatientEngine) that brings together AI-driven tools to help researchers and pharma interact seamlessly with complex biological data.

Our toolkit currently consists of three intelligent agents, each designed to simplify and enhance access to specialized data in biology:

- [**Talk2BioModels**](talk2biomodels/models/intro.md): Engage directly with mathematical models in systems biology.

- [**Talk2Cells**](talk2cells/intro.md) *(Coming soon)*: Query and analyze sequencing data with ease.

- [**Talk2KnowledgeGraphs**](Talk2KnowledgeGraphs/intro.md) *(Coming soon)*: Access and explore complex biological knowledge graphs for insightful data connections.

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Git**
- Required libraries specified in `requirements.txt`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma
   cd AIAgents4Pharma
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize OPENAI_API_KEY**
   ```bash
   export OPENAI_API_KEY = ....
   ```

4. **Launch the agent:**
   To launch the Talk2BioModels agent, run:
   ```bash
   streamlit run app/frontend/streamlit_app.py
   ```

For detailed instructions on each agent, please refer to their respective folders.

## Contributing

We welcome contributions to AIAgents4Pharma! Here’s how you can help:

1. **Fork the repository**
2. **Create a new branch** for your feature (`git checkout -b feat/feature-name`)
3. **Commit your changes** (`git commit -m 'feat: Add new feature'`)
4. **Push to the branch** (`git push origin feat/feature-name`)
5. **Open a pull request**

Check out our [CONTRIBUTING.md](CONTRIBUTING.md) for more information.


