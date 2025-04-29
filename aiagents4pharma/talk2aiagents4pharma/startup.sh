#!/bin/bash
set -e

FORCE_CPU=false

# Parse arguments
while [[ $# -gt 0 ]]; do
	case "$1" in
	--cpu)
		FORCE_CPU=true
		shift
		;;
	*)
		shift
		;;
	esac
done

echo "[STARTUP] Detecting hardware configuration..."

GPU_TYPE="cpu"
ARCH=$(uname -m)

if [ "$FORCE_CPU" = true ]; then
	echo "[STARTUP] --cpu flag detected. Forcing CPU mode."
	GPU_TYPE="cpu"

elif command -v nvidia-smi >/dev/null 2>&1; then
	echo "[STARTUP] Hardware configuration: NVIDIA GPU detected."
	GPU_TYPE="nvidia"

elif command -v lspci >/dev/null 2>&1 && lspci | grep -i amd | grep -iq vga; then
	echo "[STARTUP] Hardware configuration: AMD GPU detected."
	GPU_TYPE="amd"

elif [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
	echo "[STARTUP] Hardware configuration: Apple Silicon (arm64) detected. Metal acceleration is not available inside Docker. Running in CPU mode."
	GPU_TYPE="cpu"

else
	echo "[STARTUP] Hardware configuration: No supported GPU detected. Running in CPU mode."
	GPU_TYPE="cpu"
fi

# Select correct Ollama image
if [ "$GPU_TYPE" = "amd" ]; then
	OLLAMA_IMAGE="ollama/ollama:rocm"
else
	OLLAMA_IMAGE="ollama/ollama:latest"
fi

# Ensure Docker network exists
docker network inspect app-network >/dev/null 2>&1 || docker network create app-network

echo "[STARTUP] Using image: $OLLAMA_IMAGE"

# Start Ollama container
if [ "$GPU_TYPE" = "nvidia" ]; then
	echo "[STARTUP] Launching Ollama with NVIDIA runtime..."
	docker run -d \
		--name ollama \
		--runtime=nvidia \
		--network app-network \
		-v ollama_data:/root/.ollama \
		-p 11434:11434 \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		--entrypoint /bin/sh \
		"$OLLAMA_IMAGE" \
		-c "ollama serve & sleep 10 && ollama pull nomic-embed-text && tail -f /dev/null"

elif [ "$GPU_TYPE" = "amd" ]; then
	echo "[STARTUP] Launching Ollama with AMD ROCm..."
	docker run -d \
		--name ollama \
		--network app-network \
		-v ollama_data:/root/.ollama \
		-p 11434:11434 \
		--device=/dev/kfd \
		--device=/dev/dri \
		-e ROC_ENABLE_PRE_VEGA=1 \
		-e HSA_ENABLE_SDMA=0 \
		--entrypoint /bin/sh \
		"$OLLAMA_IMAGE" \
		-c "ollama serve & sleep 10 && ollama pull nomic-embed-text && tail -f /dev/null"

else
	echo "[STARTUP] Launching Ollama in CPU mode..."
	OLLAMA_IMAGE=$OLLAMA_IMAGE docker compose up -d ollama
fi

# Wait for Ollama API to be ready
echo "[STARTUP] Waiting for Ollama API..."
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
	echo "[STARTUP] Ollama not ready yet..."
	sleep 5
done

# Pull the model (only for CPU mode; already handled in GPU mode)
if [ "$GPU_TYPE" = "cpu" ]; then
	echo "[STARTUP] Pulling model 'nomic-embed-text' for CPU setup..."
	docker exec ollama ollama pull nomic-embed-text
fi

# Wait for model to appear
echo "[STARTUP] Waiting for model 'nomic-embed-text' to become available..."
until docker exec ollama ollama list | grep -q "nomic-embed-text"; do
	echo "[STARTUP] Model not ready yet..."
	sleep 5
done

echo "[STARTUP] Model is ready. Starting talk2aiagents4pharma agent..."
docker compose up -d talk2aiagents4pharma

echo "[STARTUP] System fully running at: http://localhost:8501"
