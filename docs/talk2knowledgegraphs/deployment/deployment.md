# 🛠️ Deployment Guide for Talk2KnowledgeGraphs (T2KG)

This step-by-step tutorial helps you deploy **Talk2KnowledgeGraphs (T2KG)** on your local machine.

> **Note:** This deployment guide assumes that you have access to a machine with **NVIDIA GPU(s)**.

---

## ✅ Step 1: Install Conda

Install the Anaconda Python distribution, which simplifies package and environment management.

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash Anaconda3-2025.06-0-Linux-x86_64.sh
source ~/.bashrc
```

---

## ✅ Step 2: Install NVIDIA CUDA Toolkit

Install NVIDIA CUDA libraries to enable GPU-accelerated computation required for model inference.

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

---

## ✅ Step 3: Install NVIDIA Container Toolkit for Docker

This allows Docker containers to access your GPU using the NVIDIA runtime.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo apt-get update
```

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

> For more details, see the [official NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.17.8/install-guide.html).

---

## ✅ Step 4: Restart Docker

Reload Docker to apply the NVIDIA runtime settings.

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

---

## ✅ Step 5: Install Python 3.12 Virtual Environment

This is optional but recommended if you're running code outside Docker and want isolated Python environments.

```bash
sudo apt install python3.12-venv
```

---

## ✅ Step 6: Get the docker-compose.yml

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

Setup environment variables

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
MILVUS_DATABASE=your_database_name_here

# Specify the data directory for multimodal data to your own data directory
# DATA_DIR=/your_absolute_path_to_your_data_dir/

BATCH_SIZE=500
```

---

## ✅ Step 8: Launch Dockerized T2KG Pipeline

This starts the backend (Milvus, API server) and frontend (Streamlit UI) in containers.

```bash
docker compose up -d
```

## 🧹 Optional: Reset and Clean Up Docker Containers

### Stop containers

```bash
sudo docker compose down -v
```

### Remove local volumes (stored graph/embedding data)

```bash
sudo rm -rf volumes
```
