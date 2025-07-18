# Talk2BioModels

## Installation

### Docker (stable-release)

_This agent is available on Docker Hub._

### Run via `docker run`

```sh
docker run -d \
  --name talk2biomodels \
  -e OPENAI_API_KEY=<your_openai_api_key> \
  -e NVIDIA_API_KEY=<your_nvidia_api_key> \
  -p 8501:8501 \
  virtualpatientengine/talk2biomodels
```

### Access the Web UI

Once started, open:

```
http://localhost:8501
```

---

## Environment Variables

- `OPENAI_API_KEY` – required
- `NVIDIA_API_KEY` – required

---

## Get Key

- `NVIDIA_API_KEY` – required (obtain a free key at [https://build.nvidia.com/explore/discover](https://build.nvidia.com/explore/discover))

**LangSmith** support is optional. To enable it, create an API key [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key).

_Please note that this will create a new tracing project in your Langsmith
account with the name `T2X-xxxx`, where `X` can be `B` (Biomodels).
If you skip the previous step, it will default to the name `default`.
`xxxx` will be the 4-digit ID created for the session._

---

## Running Multiple Agents

By default, this agent listens on port `8501`. To run another agent on a different port, map a different host port:

```sh
docker run -d \
  --name talk2biomodels \
  -e OPENAI_API_KEY=<your_openai_api_key> \
  -e NVIDIA_API_KEY=<your_nvidia_api_key> \
  -p 8502:8501 \
  virtualpatientengine/talk2biomodels
```

Then visit: [http://localhost:8502](http://localhost:8502)
