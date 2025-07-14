**Talk2KnowledgeGraphs** is an AI agent designed to interact with biomedical knowledge graphs. Biomedical knowledge graphs contains crucial information in the form of entities (nodes) and their relationships (edges). These graphs are used to represent complex biological systems, such as metabolic pathways, protein-protein interactions, and gene regulatory networks. In order to easily interact with this information, Talk2KnowledgeGraphs uses natural language processing (NLP) to enable users to ask questions and make requests. By simply asking questions or making requests, users can:

- Dataset loading: load knowledge graph from datasets.
- Embedding: embed entities and relationships in the knowledge graph.
- Knowledge graph construction: construct a knowledge graph from dataframes.
- Subgraph extraction: extract subgraphs from the initial knowledge graph.
- Retrieval: retrieve information from the (sub-) knowledge graph.
- Reasoning: reason over the (sub-) knowledge graph.
- Visualization: visualize the (sub-) knowledge graph.

## Installation

### Docker (stable-release)

_This agent is available on Docker Hub._

**Prerequisites**
- If your machine has NVIDIA GPU(s), please install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.17.8/install-guide.html) (required for GPU support with Docker; enables containers to access NVIDIA GPUs for accelerated computing). After installing `nvidia-container-toolkit`, please restart Docker to ensure GPU support is enabled.
- [Ollama](https://ollama.com/) (for embedding models like `nomic-embed-text`)

---

#### 1. Download files

```sh
mkdir talk2knowledgegraphs && cd talk2knowledgegraphs
wget https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/docker-compose.yml \
     https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/.env.example \
     https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/startup.sh
```

#### 2. Setup environment variables

```sh
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=...                  # Required for agent
NVIDIA_API_KEY=...                  # Required for embedding models
OLLAMA_HOST=http://ollama:11434     # Required for embedding models
LANGCHAIN_TRACING_V2=true           # Optional tracing
LANGCHAIN_API_KEY=...               # Optional tracing
```

---

#### 3. Start the agent

```sh
chmod +x startup.sh
./startup.sh        # Add --cpu flag to force CPU-only mode if needed
```

---

### Access the Web UI

Once started, open:

```
http://localhost:8501
```

---

## Get Key

- `NVIDIA_API_KEY` – required (obtain a free key at [https://build.nvidia.com/explore/discover](https://build.nvidia.com/explore/discover))

**LangSmith** support is optional. To enable it, create an API key [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key).

_Please note that this will create a new tracing project in your Langsmith
account with the name `T2X-xxxx`, where `X` can be `KG` (KnowledgeGraphs).
If you skip the previous step, it will default to the name `default`.
`xxxx` will be the 4-digit ID created for the session._

---

## Notes for Windows Users

If you are using Windows, it is recommended to install [**Git Bash**](https://git-scm.com/downloads) for a smoother experience when running the bash commands in this guide.

- For applications that use **Docker Compose**, Git Bash is **required**.
- For applications that use **docker run** manually, Git Bash is **optional**, but recommended for consistency.

You can download Git Bash here: [Git for Windows](https://git-scm.com/downloads).

When using Docker on Windows, make sure you **run Docker with administrative privileges** if you face permission issues.

To resolve permission issues, you can:

- Review the official Docker documentation on [Windows permission requirements](https://docs.docker.com/desktop/setup/install/windows-permission-requirements/).
- Alternatively, follow the community discussion and solutions on [Docker Community Forums](https://forums.docker.com/t/error-when-trying-to-run-windows-containers-docker-client-must-be-run-with-elevated-privileges/136619).

---

## About `startup.sh`

When executed, the script will:

- Detect NVIDIA, AMD, or CPU hardware (Apple Metal unsupported inside Docker).
- Select the appropriate Ollama image (`latest` or `rocm`).
- Launch the Ollama container with correct runtime options.
- Pull the embedding model (`nomic-embed-text`).
- Start the agent once the model is ready.

#### Option 2: PyPI *(coming soon)*
   ```bash
   pip install aiagents4pharma
   ```
