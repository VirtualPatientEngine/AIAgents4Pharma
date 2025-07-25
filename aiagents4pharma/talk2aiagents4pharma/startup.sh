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

# Clean up any existing containers first
echo "[STARTUP] Cleaning up any existing containers..."
docker rm -f talk2aiagents4pharma 2>/dev/null || true

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

# Download appropriate Milvus docker-compose file
if [ "$GPU_TYPE" = "cpu" ]; then
	echo "[STARTUP] Downloading Milvus CPU docker-compose file..."
	wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose.yml -O milvus-docker-compose.yml
else
	echo "[STARTUP] Downloading Milvus GPU docker-compose file..."
	wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose-gpu.yml -O milvus-docker-compose.yml
fi

echo "[STARTUP] Starting Milvus with $GPU_TYPE configuration..."

# Start Milvus services
docker compose -f milvus-docker-compose.yml up -d

# Wait for Milvus API to be ready
echo "[STARTUP] Waiting for Milvus API..."
MILVUS_READY=false
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
	if curl -s http://localhost:19530/health >/dev/null 2>&1; then
		echo "[STARTUP] Milvus health check passed"
		MILVUS_READY=true
		break
	else
		echo "[STARTUP] Milvus not ready yet... (attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS)"
		sleep 10
		ATTEMPT=$((ATTEMPT + 1))
	fi
done

if [ "$MILVUS_READY" = false ]; then
	echo "[STARTUP] ERROR: Milvus failed to start after $MAX_ATTEMPTS attempts"
	echo "[STARTUP] Checking Milvus logs..."
	docker logs milvus-standalone --tail 20
	echo "[STARTUP] Checking if port 19530 is in use..."
	netstat -ln | grep 19530 || lsof -i :19530 || echo "Port 19530 not found"
	exit 1
fi

# ===== DATA LOADING SECTION =====
echo "[STARTUP] Milvus is ready. Starting data loading process..."

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
	echo "[STARTUP] Loading environment variables from .env..."
	set -a
	source .env
	set +a
else
	echo "[STARTUP] WARNING: .env file not found. Using default values or environment fallback."
fi

# Warn if DATA_DIR is not set
if [ -z "${DATA_DIR}" ]; then
 echo "[STARTUP] WARNING: DATA_DIR is not set in .env. Python will use default fallback."
fi

# Check if data loading should be performed
if [ -f "../talk2knowledgegraphs/milvus_data_dump.py" ] && \
   { [ -n "${DATA_DIR}" ] && \
     [ -d "${DATA_DIR}/nodes/enrichment" ] && \
     [ -d "${DATA_DIR}/nodes/embedding" ] && \
     [ -d "${DATA_DIR}/edges/enrichment" ] && \
     [ -d "${DATA_DIR}/edges/embedding" ]; } || \
	 [ -z "${DATA_DIR}" ]; then

 	echo "[STARTUP] Proceeding with data loading (env: ${DATA_DIR:-<default used by Python>})"


	# Create virtual environment for data loading
	VENV_DIR="venv"
	echo "[STARTUP] Creating virtual environment: $VENV_DIR"
	python3.12 -m venv $VENV_DIR

	# Activate virtual environment
	source $VENV_DIR/bin/activate
	echo "[STARTUP] Virtual environment activated"

	# Function to cleanup virtual environment
	cleanup_venv() {
		echo "[STARTUP] Cleaning up virtual environment..."
		deactivate 2>/dev/null || true
		rm -rf $VENV_DIR
		echo "[STARTUP] Virtual environment cleaned up"
	}

	# Set trap to ensure cleanup happens even if script fails
	trap cleanup_venv EXIT

	# Run data loading script
	echo "[STARTUP] Running data loading script..."
	if python3.12 ../talk2knowledgegraphs/milvus_data_dump.py; then
		echo "[STARTUP] Data loading completed successfully!"
	else
		echo "[STARTUP] ERROR: Data loading failed!"
		exit 1
	fi

	# Deactivate virtual environment (cleanup will happen via trap)
	deactivate
	echo "[STARTUP] Data loading process completed"

else
	echo "[STARTUP] Data loading skipped - script or required folders not found"
	echo "[STARTUP] Expected files/folders in: $DATA_DIR"
	echo "[STARTUP]   - ../talk2knowledgegraphs/milvus_data_dump.py"
	echo "[STARTUP]   - nodes/enrichment, nodes/embedding"
	echo "[STARTUP]   - edges/enrichment, edges/embedding"
fi

# ===== END DATA LOADING SECTION =====

# Ensure Docker network exists
docker network inspect milvus >/dev/null 2>&1 || docker network create milvus

echo "[STARTUP] Starting talk2aiagents4pharma application..."

# Configure Docker Compose for talk2aiagents4pharma based on GPU type
if [ "$GPU_TYPE" = "nvidia" ]; then
	echo "[STARTUP] Configuring Docker Compose for NVIDIA GPU..."
	cat > docker-compose-gpu.yml << 'EOF'
services:
  talk2aiagents4pharma:
    platform: linux/amd64
    image: vpatientengine/talk2aiagents4pharma:latest
    container_name: talk2aiagents4pharma
    ports:
      - "8501:8501"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              device_ids: ["0"]
    environment:
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
    env_file:
      - .env
    restart: unless-stopped
    networks:
      - milvus

networks:
  milvus:
    external: true
    name: milvus
EOF
	COMPOSE_FILE="docker-compose-gpu.yml"
else
	echo "[STARTUP] Using CPU-only configuration..."
	COMPOSE_FILE="docker-compose.yml"
fi

# Start the main application with appropriate configuration
docker compose -f $COMPOSE_FILE up -d talk2aiagents4pharma

# Wait a moment for the application to start
sleep 10

# Clean up temporary files
echo "[STARTUP] Cleaning up temporary files..."
rm -f milvus-docker-compose.yml
if [ "$GPU_TYPE" = "nvidia" ]; then
	rm -f docker-compose-gpu.yml
	echo "[STARTUP] Removed docker-compose-gpu.yml"
fi
echo "[STARTUP] Removed milvus-docker-compose.yml"

echo "[STARTUP] System fully running at: http://localhost:8501"
echo "[STARTUP] Milvus API available at: http://localhost:19530"