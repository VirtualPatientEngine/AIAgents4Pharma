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

Welcome to **AIAgents4Pharma** – an open-source project by [Team VPE](https://github.com/VirtualPatientEngine) that brings together AI-driven tools to help researchers and pharma interact seamlessly with complex biological data.

Our toolkit currently consists of the following agents:

- **Talk2BioModels** _(v1 released; v2 in progress)_: Engage directly with mathematical models in systems biology.
- **Talk2KnowledgeGraphs** _(v1 in progress)_: Access and explore complex biological knowledge graphs for insightful data connections.
- **Talk2Scholars** _(v1 in progress)_: Get recommendations for articles related to your choice. Download, query, and write/retrieve them to your reference manager (currently supporting Zotero).
- **Talk2Cells** _(v1 in progress)_: Query and analyze sequencing data with ease.
- **Talk2AIAgents4Pharma** _(v1 in progress)_: Converse with all the agents above (currently supports T2B and T2KG)

![AIAgents4Pharma](assets/AIAgents4Pharma.png)

## Getting Started

### Installation

#### Option 1: Docker (stable-release)

_We now have all the agents available on Docker Hub._

##### **To run Talk2AIAgents4Pharma / Talk2KnowledgeGraphs**

Both agents require [Ollama](https://ollama.com/) to run embedding models like `nomic-embed-text`. We use a **single startup script** that automatically detects your hardware (NVIDIA, AMD, or CPU) and handles container startup, model loading, and service orchestration.

##### **1. Clone the repository and navigate to the agent directory**

```sh
git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma

cd AIAgents4Pharma/aiagents4pharma/<agent>
```

Replace `<agent>` with either:

- `talk2aiagents4pharma`
- `talk2knowledgegraphs`

##### **2. Setup environment variables**

Copy and configure your `.env` file:

```sh
cp .env.example .env
```

Then edit `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
NVIDIA_API_KEY=your_nvidia_api_key
```

_Notes:_

- `OPENAI_API_KEY` is required for both agents.
- `NVIDIA_API_KEY` is needed only for Talk2AIAgents4Pharma. ([Click here for free credits](https://build.nvidia.com/explore/discover))
- `OLLAMA_HOST` is already preconfigured.
- LangSmith support is optional and can be enabled in `.env`.

##### **3. Start the application**

Run the startup script. It will:

- Detect your GPU (NVIDIA, AMD, or CPU)
- Choose the correct Ollama image (`latest` or `rocm`)
- Launch the Ollama container with appropriate runtime settings
- Pull the required embedding model (`nomic-embed-text`)
- Start the agent **after the model is available**

```sh
chmod +x startup.sh
./startup.sh
```

##### **4. Access the Web UI**

Once started, the agent is available at:

```
http://localhost:8501
```

##### **5. Stop or reset containers**

To stop all services:

```sh
docker compose down
```

To stop and delete volumes (e.g., model cache):

```sh
docker compose down -v
```

---

##### **To Run Talk2Biomodels / Talk2Scholars**

1. **Run the containers**

###### talk2biomodels

```bash
docker run -d \
  --name talk2biomodels \
  -e OPENAI_API_KEY=<your_openai_api_key> \
  -e NVIDIA_API_KEY=<your_nvidia_api_key> \
  -p 8501:8501 \
  virtualpatientengine/talk2biomodels
```

###### talk2scholars

```bash
docker run -d \
  --name talk2scholars \
  -e OPENAI_API_KEY=<your_openai_api_key> \
  -e ZOTERO_API_KEY=<your_zotero_api_key> \
  -e ZOTERO_USER_ID=<your_zotero_user_id> \
  -p 8501:8501 \
  virtualpatientengine/talk2scholars
```

2. **Access the Web App**
   Open your browser and go to:

   ```
   http://localhost:8501
   ```

   _You can create a free NVIDIA account and apply for free API credits [here](https://build.nvidia.com/explore/discover)._  
    _You can generate your Zotero API keys [here](https://www.zotero.org/user/login#applications)._
   _Please note that ZOTERO keys are requried only if you want to launch Talk2Scholars. For all the other agents, please ignore this step._

#### Notes

- Be sure to **replace the placeholder values** with your actual credentials before running any container:

  - `<your_openai_api_key>`
  - `<your_nvidia_api_key>`
  - `<your_zotero_api_key>`
  - `<your_zotero_user_id>`

- All agents default to **port `8501`**. If you plan to run multiple agents simultaneously, make sure to assign **different ports** to avoid conflicts.

  Example (Talk2Scholars on port `8502`):

  ```bash
  docker run -d \
    --name talk2scholars \
    -e OPENAI_API_KEY=<your_openai_api_key> \
    -e ZOTERO_API_KEY=<your_zotero_api_key> \
    -e ZOTERO_USER_ID=<your_zotero_user_id> \
    -p 8502:8501 \
    virtualpatientengine/talk2scholars
  ```

  Then access the app at: [http://localhost:8502](http://localhost:8502)

---

#### Option 2: git (for developers and contributors)

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma
   cd AIAgents4Pharma
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   ⚠️ The current version of T2KG requires additional Ollama library to be installed.

   Ollama can be easily downloaded and installed from the following [link](https://ollama.com/download)

   As an alternative, use the following commands to install the library using terminal and to pull necessary model:

   - Ubuntu:
     ```
     curl -fsSL https://ollama.com/install.sh | sh
     ollama pull nomic-embed-text
     ```
   - Windows:
     ```
     curl -L https://ollama.com/download/ollama-windows-amd64.zip -o ollama-windows-amd64.zip
     tar -xzf .\ollama-windows-amd64.zip
     start ollama serve
     ollama pull nomic-embed-text
     ```
   - macOS:
     ```
     brew install ollama
     ollama pull nomic-embed-text
     ```
     A list of pulled Ollama models can be checked using the following command:

   ```
   ollama list
   ```

   ⚠️ `pcst_fast 1.0.10` library requires `Microsoft Visual C++ 14.0` or greater to be installed.
   You can download `Microsoft C++ Build Tools` from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

3. **Initialize API Keys**

   ```bash
   export OPENAI_API_KEY=....
   export NVIDIA_API_KEY=....
   export ZOTERO_API_KEY=....
   export ZOTERO_USER_ID=....
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY=...
   ```

   _You can create a free NVIDIA account and apply for free API credits [here](https://build.nvidia.com/explore/discover)._  
   _You can generate your Zotero API keys [here](https://www.zotero.org/user/login#applications)._  
   _Please note that ZOTERO keys are requried only if you want to launch Talk2Scholars. For all the other agents, please ignore this step._  
   _LangSmith support is optional, you can get the api keys [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key$0)_

   _Please note that this will create a new tracing project in your Langsmith
   account with the name `T2X-xxxx`, where `X` can be `AA4P` (Main Agent),
   `B` (Biomodels), `S` (Scholars), `KG` (KnowledgeGraphs), or `C` (Cells).
   If you skip the previous step, it will default to the name `default`.
   `xxxx` will be the 4-digit ID created for the session._

4. **Launch the app:**
   ```bash
   streamlit run app/frontend/streamlit_app_<agent>.py
   ```
   _Replace `<agent>` with the agent name you are interested to launch:_

- `talk2aiagents4pharma`
- `talk2knowledgegraphs`
- `talk2biomodels`
- `talk2scholars`
- `talk2cells`

For detailed instructions on each agent, please refer to their respective modules.

---

#### Option 3: pip (beta-release)

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

```bash
pip install aiagents4pharma
```

Check out the tutorials on each agent for detailed instructions.

---

## Contributing

We welcome contributions to AIAgents4Pharma! Here’s how you can help:

1. **Fork the repository**
2. **Create a new branch** for your feature (`git checkout -b feat/feature-name`)
3. **Commit your changes** (`git commit -m 'feat: Add new feature'`)
4. **Push to the branch** (`git push origin feat/feature-name`)
5. **Open a pull request** and reach out to any one of us below via Discussions:

   _Note: We welcome all contributions, not just programming-related ones. Feel free to open bug reports, suggest new features, or participate as a beta tester. Your support is greatly appreciated!_

- **Talk2Biomodels/Talk2Cells**: [@gurdeep330](https://github.com/gurdeep330) [@lilijap](https://github.com/lilijap) [@dmccloskey](https://github.com/dmccloskey)
- **Talk2KnowledgeGraphs**: [@awmulyadi](https://github.com/awmulyadi) [@dmccloskey](https://github.com/dmccloskey)
- **Talk2Scholars**: [@ansh-info](https://github.com/ansh-info) [@gurdeep330](https://github.com/gurdeep330) [@dmccloskey](https://github.com/dmccloskey)

### Current Needs

- **Beta testers** for Talk2BioModels and Talk2Scholars.
- **Developers** with experience in Python and Bioinformatics and/or knowledge graphs for contributions to AIAgents4Pharma.

Feel free to reach out to us via Discussions.

Check out our [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

---

## Feedback

Questions/Bug reports/Feature requests/Comments/Suggestions? We welcome all. Please use `Issues` or `Discussions` 😀
