# Talk2Scholars

## Installation

### Docker (stable-release)

**Prerequisites**

- [Milvus](https://milvus.io) (for a vector database)

---

#### 1. Download files

```sh
mkdir talk2scholars && cd talk2scholars
wget https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2scholars/docker-compose.yml \
     https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2scholars/.env.example \
     https://raw.githubusercontent.com/VirtualPatientEngine/AIAgents4Pharma/main/aiagents4pharma/talk2scholars/startup.sh
```

#### 2. Setup environment variables

```sh
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=...                  # Required
NVIDIA_API_KEY=...                  # Required
ZOTERO_API_KEY=...                  # Required
ZOTERO_USER_ID=...                  # Required
MILVUS_HOST=milvus-standalone       # Required
MILVUS_PORT=19530                   # Required
MILVUS_DB_NAME=...                  # Required
MILVUS_COLLECTION_NAME=...          # Required
LANGCHAIN_TRACING_V2=true           # Optional
LANGCHAIN_API_KEY=...               # Optional
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
- `ZOTERO_API_KEY` – required (generate at [https://www.zotero.org/user/login#applications](https://www.zotero.org/user/login#applications))

**LangSmith** support is optional. To enable it, create an API key [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key).

_Please note that this will create a new tracing project in your Langsmith
account with the name `T2X-xxxx`, where `X` can be `S` (Scholars).
If you skip the previous step, it will default to the name `default`.
`xxxx` will be the 4-digit ID created for the session._

---

## Notes for Windows Users

If you are using Windows, it is recommended to install [**Git Bash**](https://git-scm.com/downloads) for a smoother experience when running the bash commands in this guide.

- For applications that use **Docker Compose**, Git Bash is **required**.
- For applications that use **docker run** manually, Git Bash is **optional**, but recommended for consistency.

You can download Git Bash here: [Git for Windows](https://git-scm.com/downloads).

When using Docker on Windows, make sure you **run Docker with administrative privileges** if you face permission issues.

To resolve for permission issues, you can:

- Review the official Docker documentation on [Windows permission requirements](https://docs.docker.com/desktop/setup/install/windows-permission-requirements/).
- Alternatively, follow the community discussion and solutions on [Docker Community Forums](https://forums.docker.com/t/error-when-trying-to-run-windows-containers-docker-client-must-be-run-with-elevated-privileges/136619).

---

## About `startup.sh`

Run the startup script. It will:

- Detect your hardware configuration (NVIDIA GPU, AMD GPU, or CPU). Apple Metal is unavailable inside Docker, and Intel SIMD optimizations are automatically handled without special configuration.
- Choose the correct Milvus image (`CPU` or `GPU`).
- Launch the Milvus container with appropriate runtime settings.
- Start the agent after the model is available.
