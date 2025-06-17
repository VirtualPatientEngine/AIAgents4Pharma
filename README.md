[![Talk2BioModels](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2biomodels.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2biomodels.yml)
[![Talk2Cells](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2cells.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2cells.yml)
[![Talk2KnowledgeGraphs](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2knowledgegraphs.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2knowledgegraphs.yml)
[![TESTS Talk2Scholars](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2scholars.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2scholars.yml)
[![TESTS Talk2AIAgents4Pharma](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2aiagents4pharma.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2aiagents4pharma.yml)
![GitHub Release](https://img.shields.io/github/v/release/VirtualPatientEngine/AIAgents4Pharma)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![Talk2AIAgents4Pharma Pulls](https://img.shields.io/docker/pulls/virtualpatientengine/talk2aiagents4pharma?label=Talk2AIAgents4Pharma%20Pulls&color=blue&logo=docker&style=flat-square)
![Talk2Scholars Pulls](https://img.shields.io/docker/pulls/virtualpatientengine/talk2scholars?label=Talk2Scholars%20Pulls&color=blue&logo=docker&style=flat-square)
![Talk2BioModels Pulls](https://img.shields.io/docker/pulls/virtualpatientengine/talk2biomodels?label=Talk2BioModels%20Pulls&color=blue&logo=docker&style=flat-square)
![Talk2KnowledgeGraphs Pulls](https://img.shields.io/docker/pulls/virtualpatientengine/talk2knowledgegraphs?label=Talk2KnowledgeGraphs%20Pulls&color=blue&logo=docker&style=flat-square)

## Introduction

Welcome to **AIAgents4Pharma** – an open-source project by [Team VPE](https://bmedx.com/research-teams/artificial-intelligence/team-vpe/) that brings together AI-driven tools to help researchers and pharma interact seamlessly with complex biological data.

Our toolkit currently consists of the following agents:

- **Talk2BioModels** _(v1 released; v2 in progress)_: Engage directly with mathematical models in systems biology.
- **Talk2KnowledgeGraphs** _(v1 in progress)_: Access and explore complex biological knowledge graphs for insightful data connections.
- **Talk2Scholars** _(v1 in progress)_: Get recommendations for articles related to your choice. Download, query, and write/retrieve them to your reference manager (currently supporting Zotero).
- **Talk2Cells** _(v1 in progress)_: Query and analyze sequencing data with ease.
- **Talk2AIAgents4Pharma** _(v1 in progress)_: Converse with all the agents above (currently supports T2B and T2KG)

![AIAgents4Pharma](docs/assets/AIAgents4Pharma.png)

## News

- T2B and T2KG accepted at the MLGenX workshop during ICLR #2025 in Singapore. [Read More](https://openreview.net/forum?id=av4QhBNeZo)

<div align="center">
<strong>Watch the presentation:</strong><br><br>
<a href="https://www.youtube.com/watch?v=3cU_OxY4HiE">
<img src="https://img.youtube.com/vi/3cU_OxY4HiE/0.jpg" alt="Watch the presentation" width="480">
</a>
</div>

## Getting Started

### Installation

#### Option 1: Docker (stable-release)

_We now have all the agents available on Docker Hub._

##### **To run Talk2AIAgents4Pharma / Talk2KnowledgeGraphs**

Both agents require [Ollama](https://ollama.com/) to run embedding models like `nomic-embed-text`. We use a **single startup script** that automatically detects your hardware (NVIDIA, AMD, or CPU) and handles container startup, model loading, and service orchestration.

##### **1. Download docker-compose.yml, .env.example and startup.sh from GitHub**

###### Talk2Agents4Pharma

```sh
mkdir talk2aiagents4pharma && cd talk2aiagents4pharma && wget https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2aiagents4pharma/docker-compose.yml https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2aiagents4pharma/.env.example https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2aiagents4pharma/startup.sh
```

###### Talk2KnowledgeGraphs

```sh
mkdir talk2knowledgegraphs && cd talk2knowledgegraphs && wget https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/docker-compose.yml https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/.env.example https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/startup.sh
```

##### **2. Setup environment variables**

Copy and configure your `.env` file:

```sh
cp .env.example .env
```

Then edit `.env` and add your API keys:

```env
OPENAI_API_KEY=...                  # Required for both agents
NVIDIA_API_KEY=...                  # Required for both agents
OLLAMA_HOST=http://ollama:11434     # Required for AA4P / T2KG
LANGCHAIN_TRACING_V2=true           # Optional for both agents
LANGCHAIN_API_KEY=...               # Optional for both agents
```

[Additional Notes for Windows Users](https://github.com/VirtualPatientEngine/AIAgents4Pharma/blob/main/aiagents4pharma/talk2aiagents4pharma/install.md#notes-for-windows-users)

##### **3. Start the application**

```sh
chmod +x startup.sh
./startup.sh        # Add --cpu flag to force CPU mode if needed
```

[More about startup script](https://github.com/VirtualPatientEngine/AIAgents4Pharma/blob/main/aiagents4pharma/talk2aiagents4pharma/install.md#about-startupsh)

##### **To Run Talk2Biomodels / Talk2Scholars**

###### Talk2Biomodels

```docker
docker run -d \
--name talk2biomodels \
-e OPENAI_API_KEY=<your_openai_api_key> \
-e NVIDIA_API_KEY=<your_nvidia_api_key> \
-p 8501:8501 \
virtualpatientengine/talk2biomodels
```

###### Talk2Scholars

```docker
docker run -d \
--name talk2scholars \
-e OPENAI_API_KEY=<your_openai_api_key> \
-e ZOTERO_API_KEY=<your_zotero_api_key> \
-e ZOTERO_USER_ID=<your_zotero_user_id> \
-e NVIDIA_API_KEY=<your_nvidia_api_key> \
-p 8501:8501 \
virtualpatientengine/talk2scholars
```

##### **4. Access the Web UI**

Once started, the agent is available at:

```
http://localhost:8501
```

To use **Talk2AIAgents4Pharma** or **Talk2KnowledgeGraphs**, you need a free **NVIDIA API key**. Create an account and apply for free credits [here](https://build.nvidia.com/explore/discover).

To use **Talk2BioModels** or **Talk2Scholars**, you need a free **NVIDIA API key**. Create an account and apply for free credits [here](https://build.nvidia.com/explore/discover).

Only for **Talk2Scholars**, you also need a **Zotero API key**, which you can generate [here](https://www.zotero.org/user/login#applications). _(For all other agents, the Zotero key is not required.)_

If you are using docker on Windows, please follow these [Windows Setup Notes](https://github.com/VirtualPatientEngine/AIAgents4Pharma/blob/main/aiagents4pharma/talk2aiagents4pharma/install.md#notes-for-windows-users).

**LangSmith** support is optional. To enable it, create an API key [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key).

[More on running multiple agents simultaneously](https://github.com/VirtualPatientEngine/AIAgents4Pharma/blob/main/aiagents4pharma/talk2aiagents4pharma/install.md#to-run-multiple-agents-simultaneously)

#### Option 2: git (for developers and contributors)

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

1. **Clone the repository:**

```sh
git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma && cd AIAgents4Pharma
```

2. **Install dependencies:**

We use Conda as our environment manager, Follow the official [Quickstart](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) install instructions provided by anaconda/miniconda.

```python
conda create --name AIAgents4Pharma python=3.12 -y && conda activate AIAgents4Pharma && pip install --upgrade pip && pip install -r requirements.txt
```

3. **Initialize API Keys**

```env
export OPENAI_API_KEY=....          # Required for all agents
export NVIDIA_API_KEY=....          # Required for all agents
export ZOTERO_API_KEY=....          # Required for T2S
export ZOTERO_USER_ID=....          # Required for T2S
export LANGCHAIN_TRACING_V2=true    # Optional for all agents
export LANGCHAIN_API_KEY=...        # Optional for all agents
```

To use **Talk2AIAgents4Pharma**, **Talk2BioModels**, **Talk2KnowledgeGraphs**, or **Talk2Scholars**, you need a free **NVIDIA API key**. Create an account and apply for free credits [here](https://build.nvidia.com/explore/discover).

Only for **Talk2Scholars**, you also need a **Zotero API key**, which you can generate [here](https://www.zotero.org/user/login#applications). _(For all other agents, the Zotero key is not required.)_

To use **Talk2Scholars**, you must have **FAISS** installed through **Conda**. Follow installation instructions for your OS [here](https://github.com/VirtualPatientEngine/AIAgents4Pharma/tree/main/aiagents4pharma/talk2scholars/install.md).

To use **Talk2AIAgents4Pharma** or **Talk2KnowledgeGraphs**, you must have **Ollama** installed. Follow installation instructions for your OS [here](https://ollama.com/download).

After installing, pull the `nomic-embed-text` model and start the server by running:

```sh
ollama pull nomic-embed-text && ollama serve
```

More details about the model are available [here](https://ollama.com/library/nomic-embed-text).

Additionally on **Windows**, the `pcst_fast 1.0.10` library requires **Microsoft Visual C++ 14.0 or greater**.  
You can download the **Microsoft C++ Build Tools** [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

**LangSmith** support is optional. To enable it, create an API key [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key).

_Please note that this will create a new tracing project in your Langsmith
account with the name `T2X-xxxx`, where `X` can be `AA4P` (Main Agent),
`B` (Biomodels), `S` (Scholars), `KG` (KnowledgeGraphs), or `C` (Cells).
If you skip the previous step, it will default to the name `default`.
`xxxx` will be the 4-digit ID created for the session._

4. **Launch the app:**

```sh
streamlit run app/frontend/streamlit_app_<agent>.py
```

_Replace `<agent>` with the agent name you are interested to launch:_

- `talk2aiagents4pharma`
- `talk2biomodels`
- `talk2knowledgegraphs`
- `talk2scholars`
- `talk2cells`

For detailed instructions on each agent, please refer to their respective modules.

#### Option 3: pip (beta-release)

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

```sh
pip install aiagents4pharma
```

Check out the tutorials on each agent for detailed instructions.

## Contributing

We welcome your support to make **AIAgents4Pharma** even better.  
All types of contributions are appreciated — whether you're fixing bugs, adding features, improving documentation, or helping with testing, every contribution is valuable.

#### How to contribute

1. Star this repository to show your support.
2. Fork the repository.
3. Create a new branch for your work:

```sh
git checkout -b feat/your-feature-name
```

4. Make your changes and commit them:

```sh
git commit -m "feat: add a brief description of your change"
```

5. Push your branch:

```sh
git push origin feat/your-feature-name
```

6. Open a Pull Request.

#### Areas where you can help

- Beta testing for Talk2BioModels and Talk2Scholars.
- Development work related to Python, bioinformatics, or knowledge graphs.

#### Contacts for contributions

- **Talk2Biomodels**: [@lilijap](https://github.com/lilijap), [@gurdeep330](https://github.com/gurdeep330)
- **Talk2Cells**: [@gurdeep330](https://github.com/gurdeep330)
- **Talk2KnowledgeGraphs**: [@awmulyadi](https://github.com/awmulyadi)
- **Talk2Scholars**: [@ansh-info](https://github.com/ansh-info), [@gurdeep330](https://github.com/gurdeep330)

Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more detailed contribution guidelines.

## Feedback

If you have questions, bug reports, feature requests, comments, or suggestions, we would love to hear from you.  
Please open an `issue` or start a `discussion`
