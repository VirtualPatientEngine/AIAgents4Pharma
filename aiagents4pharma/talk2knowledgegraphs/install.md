**Talk2KnowledgeGraphs** is an AI agent designed to interact with biomedical knowledge graphs. Biomedical knowledge graphs contains crucial information in the form of entities (nodes) and their relationships (edges). These graphs are used to represent complex biological systems, such as metabolic pathways, protein-protein interactions, and gene regulatory networks. In order to easily interact with this information, Talk2KnowledgeGraphs uses natural language processing (NLP) to enable users to ask questions and make requests. By simply asking questions or making requests, users can:

- Dataset loading: load knowledge graph from datasets.
- Embedding: embed entities and relationships in the knowledge graph.
- Knowledge graph construction: construct a knowledge graph from dataframes.
- Subgraph extraction: extract subgraphs from the initial knowledge graph.
- Retrieval: retrieve information from the (sub-) knowledge graph.
- Reasoning: reason over the (sub-) knowledge graph.
- Visualization: visualize the (sub-) knowledge graph.

📝 By default, `talk2knowledgegraphs` includes a small subset of the PrimeKG knowledge graph, allowing users to start interacting with it out of the box.
To switch to a different knowledge graph or use your own, refer to the [deployment guide](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2knowledgegraphs/deployment/).

## Installation

If your machine has NVIDIA GPU(s), please install the following this:

- [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.17.8/install-guide.html) (required for GPU support with Docker; enables containers to access NVIDIA GPUs for accelerated computing). After installing `nvidia-container-toolkit`, please restart Docker to ensure GPU support is enabled.

### Docker (stable-release)

_This agent is available on Docker Hub._

---

#### 1. Download files

Choose the appropriate version of the `docker-compose.yml` file based on your system:

**For GPU:**

```sh
wget https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/docker-compose/gpu/docker-compose.yml \
     https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/docker-compose/gpu/.env.example
```

**For CPU:**

```sh
wget https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/docker-compose/cpu/docker-compose.yml \
     https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2knowledgegraphs/docker-compose/cpu/.env.example
```

#### 2. Setup environment variables

```sh
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# .env.example (DO NOT put actual API keys here, read the README.md)

# OPENAI API KEY
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith API KEY
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key_here

# NVIDIA API KEY
NVIDIA_API_KEY=your_nvidia_api_key_here

# Set environment variables for data loader
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=Milvus
MILVUS_DATABASE=t2kg_primekg

# Specify the data directory for multimodal data to your own data directory
# DATA_DIR=/your_absolute_path_to_your_data_dir/

BATCH_SIZE=500
```

---

#### 3. Start the agent

```sh
docker compose up -d
```

---

### Access the Web UI

Once started, open:

```
http://localhost:8501
```

> In the background, the BioBridge multimodal embeddings will be inserted into the Milvus database, and the `talk2knowledgegraphs` service will start. Once the data is fully inserted, the application will be in a healthy state and accessible at the above address.
>
> You can monitor the process using:
>
> ```sh
> docker logs -f talk2knowledgegraphs
> ```

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
