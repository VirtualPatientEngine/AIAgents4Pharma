<img src="docs/assets/VPE.png" alt="Virtual Patient Engine Logo" width="150"/>

<!--  Project Info -->

![RELEASE](https://img.shields.io/github/v/release/VirtualPatientEngine/AIAgents4Pharma?label=RELEASE)
![Docker Compose Release Version](https://img.shields.io/github/v/release/VirtualPatientEngine/AIAgents4Pharma?label=Docker%20Compose%20Version)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

<!--  Deployment Workflows -->

[![Pages Deployment](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/pages/pages-build-deployment)
[![MkDocs Deploy](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/mkdocs_deploy.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/mkdocs_deploy.yml)
[![Docker Build & Push](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/docker_build.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/docker_build.yml)
[![Docker Compose Release](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/docker_compose_release.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/docker_compose_release.yml)

<!--  Tests -->

[![TESTS Talk2AIAgents4Pharma](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2aiagents4pharma.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2aiagents4pharma.yml)
[![Talk2BioModels](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2biomodels.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2biomodels.yml)
[![Talk2KnowledgeGraphs](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2knowledgegraphs.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2knowledgegraphs.yml)
[![TESTS Talk2Scholars](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2scholars.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2scholars.yml)
[![Talk2Cells](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2cells.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/tests_talk2cells.yml)

<!--  Docker Pulls -->

![Talk2AIAgents4Pharma Pulls](https://img.shields.io/docker/pulls/vpatientengine/talk2aiagents4pharma?label=Talk2AIAgents4Pharma%20Pulls&color=blue&logo=docker&style=flat-square)
![Talk2BioModels Pulls](https://img.shields.io/docker/pulls/vpatientengine/talk2biomodels?label=Talk2BioModels%20Pulls&color=blue&logo=docker&style=flat-square)
![Talk2KnowledgeGraphs Pulls](https://img.shields.io/docker/pulls/vpatientengine/talk2knowledgegraphs?label=Talk2KnowledgeGraphs%20Pulls&color=blue&logo=docker&style=flat-square)
![Talk2Scholars Pulls](https://img.shields.io/docker/pulls/vpatientengine/talk2scholars?label=Talk2Scholars%20Pulls&color=blue&logo=docker&style=flat-square)

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

Choose your agent below for detailed Docker instructions:

- [Talk2AIAgents4Pharma](aiagents4pharma/talk2aiagents4pharma/install.md)
- [Talk2KnowledgeGraphs](aiagents4pharma/talk2knowledgegraphs/install.md)
- [Talk2BioModels](aiagents4pharma/talk2biomodels/install.md)
- [Talk2Scholars](aiagents4pharma/talk2scholars/install.md)

#### Option 2: git (for developers and contributors)

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

1. **Clone the repository:**

```sh
git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma && cd AIAgents4Pharma
```

2. **Install dependencies:**

We use `uv` for fast and reliable dependency management. Install uv first following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

> **For developers**: See [docs/developer/README.md](docs/developer/README.md) for detailed setup instructions including system prerequisites.

```sh
uv sync --extra dev --frozen
```

> 💡 **Recommended**: Use `--frozen` flag to ensure exact reproducible builds using the pinned versions from `uv.lock`.

3. **Initialize API Keys**

```env
export OPENAI_API_KEY=....          # Required for all agents
export NVIDIA_API_KEY=....          # Required for all agents
export ZOTERO_API_KEY=....          # Required for T2S
export ZOTERO_USER_ID=....          # Required for T2S
export LANGCHAIN_TRACING_V2=true    # Optional for all agents
export LANGCHAIN_API_KEY=...        # Optional for all agents
```

4. **Launch the app:**

> System Dependency: libmagic (for secure uploads)
> For accurate file MIME-type detection used by our secure upload validation, install the libmagic system library. This is recommended across all providers (OpenAI, Azure OpenAI, NVIDIA) because it runs locally in the Streamlit apps.
>
> - Linux (Debian/Ubuntu): `sudo apt-get install libmagic1`
> - macOS (Homebrew): `brew install libmagic`
> - Windows: Use the `python-magic`/`python-magic-bin` package; libmagic is bundled
>   If libmagic is not available, the apps fall back to extension-based detection. For best security, keep libmagic installed.

**Option A: Using UV (recommended)**

```sh
uv run streamlit run app/frontend/streamlit_app_<agent>.py
```

**Option B: Traditional approach**

```sh
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Then run the app
streamlit run app/frontend/streamlit_app_<agent>.py
```

_Replace `<agent>` with the agent name you are interested to launch:_

- `talk2aiagents4pharma`
- `talk2biomodels`
- `talk2knowledgegraphs`
- `talk2scholars`
- `talk2cells`

If your machine has NVIDIA GPU(s), please install the following this:

- [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.17.8/install-guide.html) (required for GPU support with Docker; enables containers to access NVIDIA GPUs for accelerated computing). After installing `nvidia-container-toolkit`, please restart Docker to ensure GPU support is enabled.

To use the **Agents**, you need a free **NVIDIA API key**. Create an account and apply for free credits [here](https://build.nvidia.com/explore/discover).

**Talk2Biomodels** supports integration with multiple LLMs: gpt-4o-mini (via OpenAI API) and open-source llama (3.1 and 3.3) models (via NVIDIA API). An **OpenAI API** key may be generated [here](https://platform.openai.com/settings/organization/api-keys). OpenAI may provide initial free credits for API calls with the API key, after which additional credits may be purchased [here](https://platform.openai.com/settings/organization/billing). More information on pricing is available [here](https://openai.com/api/pricing/).

**Talk2Scholars** and **Talk2KnowledgeGraphs** requires Milvus to be set up as the vector database — install Milvus depending on your setup by following the official instructions for [CPU](https://milvus.io/docs/install_standalone-docker-compose.md) or [GPU](https://milvus.io/docs/install_standalone-docker-compose-gpu.md). You will also need a **Zotero API key**, which you can generate [here](https://www.zotero.org/user/login#applications). _(The Zotero key is only required for Talk2Scholars; all other agents do not need it.)_

> By default, `talk2knowledgegraphs` includes a small subset of the PrimeKG knowledge graph, allowing users to start interacting with it out of the box.
> To switch to a different knowledge graph or use your own, refer to the [deployment guide](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2knowledgegraphs/deployment/).
> Additionally on **Windows**, the `pcst_fast 1.0.10` library requires **Microsoft Visual C++ 14.0 or greater**.
> You can download the **Microsoft C++ Build Tools** [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

**LangSmith** support is optional. To enable it, create an API key [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key).

_Please note that this will create a new tracing project in your Langsmith
account with the name `T2X-xxxx`, where `X` can be `AA4P` (Main Agent),
`B` (Biomodels), `S` (Scholars), `KG` (KnowledgeGraphs), or `C` (Cells).
If you skip the previous step, it will default to the name `default`.
`xxxx` will be the 4-digit ID created for the session._

#### Option 3: pip (beta-release)

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FVirtualPatientEngine%2FAIAgents4Pharma%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

```sh
pip install aiagents4pharma
```

Check out the tutorials on each agent for detailed instructions.

## Contributing

We welcome your support to make **AIAgents4Pharma** even better.
All types of contributions are appreciated — whether you're fixing bugs, adding features, improving documentation, or helping with testing, every contribution is valuable.

#### Development Setup

For contributors and developers, we have comprehensive documentation:

- **[Developer Setup Guide](docs/developer/README.md)** - Complete setup instructions with UV, security implementation, and tooling
- **[Testing & Linting Guide](docs/developer/TESTING_LINTING.md)** - How to run tests, coverage, and code quality checks
- **[SonarCloud Integration](docs/developer/SONARCLOUD_SETUP.md)** - Code quality analysis and CI/CD integration
- **[GitHub Workflows](docs/developer/WORKFLOWS.md)** - Understanding our CI/CD pipeline
- **[Streamlit Security](docs/developer/STREAMLIT_SECURITY.md)** - File upload security implementation
- **[Azure Deployment](developer/AZURE_DEPLOYMENT.md)** - Understanding our Azure deployment setup

#### How to contribute

1. Star this repository to show your support.
2. Fork the repository.
3. Create a new branch for your work:

```sh
git checkout -b feat/your-feature-name
```

4. Set up your development environment:

```sh
uv sync --extra dev --frozen  # Install development dependencies
uv run pre-commit install    # Set up code quality hooks
```

5. Make your changes and run quality checks:

```sh
uv run ruff check --fix .  # Lint and fix code
uv run ruff format .  # Format code
uv run pre-commit run --all-files  # Run all checks (linting, formatting, security)

# Run submodule-specific checks (pylint configuration in pyproject.toml)
uv run pylint aiagents4pharma/talk2scholars/
uv run coverage run --include="aiagents4pharma/talk2scholars/*" -m pytest --cache-clear aiagents4pharma/talk2scholars/tests/ && uv run coverage report
```

6. Commit and push your changes:

```sh
git commit -m "feat: add a brief description of your change"
git push origin feat/your-feature-name
```

7. Open a Pull Request.

#### Areas where you can help

- Beta testing for Talk2BioModels and Talk2Scholars.
- Development work related to Python, bioinformatics, or knowledge graphs.

#### Contacts for contributions

- **Talk2Biomodels**: [@lilijap](https://github.com/lilijap), [@gurdeep330](https://github.com/gurdeep330)
- **Talk2Cells**: [@tAndreaniSanofi](https://github.com/tAndreaniSanofi), [@gurdeep330](https://github.com/gurdeep330)
- **Talk2KnowledgeGraphs**: [@awmulyadi](https://github.com/awmulyadi)
- **Talk2Scholars**: [@ansh-info](https://github.com/ansh-info), [@gurdeep330](https://github.com/gurdeep330)

Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) and [developer documentation](docs/developer/) for detailed contribution guidelines and setup instructions.

## Feedback

If you have questions, bug reports, feature requests, comments, or suggestions, we would love to hear from you.
Please open an `issue` or start a `discussion`
