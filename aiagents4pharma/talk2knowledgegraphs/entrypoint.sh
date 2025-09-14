#!/bin/bash
# entrypoint.sh - Container entrypoint with automatic data loading

set -e

# Function to log with timestamp
log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ENTRYPOINT] $1"
}

log "=== talk2knowledgegraphs Container Starting ==="
log "Container hostname: $(hostname)"

# Show GPU info if available
if command -v nvidia-smi >/dev/null 2>&1; then
	log "Available GPUs:"
	nvidia-smi -L 2>/dev/null || log "nvidia-smi failed"
else
	log "nvidia-smi not available"
fi

# Set default values for data loader environment variables
export MILVUS_HOST=${MILVUS_HOST:-milvus-standalone}
export MILVUS_PORT=${MILVUS_PORT:-19530}
export MILVUS_USER=${MILVUS_USER:-root}
export MILVUS_PASSWORD=${MILVUS_PASSWORD:-Milvus}
export MILVUS_DATABASE=${MILVUS_DATABASE:-t2kg_primekg}
export BATCH_SIZE=${BATCH_SIZE:-500}
export CHUNK_SIZE=${CHUNK_SIZE:-5}
export AUTO_INSTALL_PACKAGES=${AUTO_INSTALL_PACKAGES:-true}
export FORCE_CPU=${FORCE_CPU:-false}
export RUN_DATA_LOADER=${RUN_DATA_LOADER:-true}

# Determine data directory with priority:
# 1. External mount (/mnt/external_data)
# 2. Environment variable DATA_DIR
# 3. Default internal path
if [ -d "/mnt/external_data" ] && [ "$(ls -A /mnt/external_data 2>/dev/null)" ]; then
	export DATA_DIR="/mnt/external_data"
	log "Using external data directory: $DATA_DIR"
elif [ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ]; then
	log "Using specified data directory: $DATA_DIR"
elif [ -d "/app/aiagents4pharma/talk2knowledgegraphs/tests/files/biobridge_multimodal/" ]; then
	export DATA_DIR="/app/aiagents4pharma/talk2knowledgegraphs/tests/files/biobridge_multimodal/"
	log "Using default internal data directory: $DATA_DIR"
else
	log "WARNING: No valid data directory found!"
	log "Checked:"
	log "  - External mount: /mnt/external_data"
	log "  - Environment DATA_DIR: ${DATA_DIR:-not set}"
	log "  - Default internal: /app/aiagents4pharma/talk2knowledgegraphs/tests/files/biobridge_multimodal/"
	log "Continuing without data loading..."
	export RUN_DATA_LOADER="false"
fi

# Display configuration
log "=== Configuration ==="
log "MILVUS_HOST: $MILVUS_HOST"
log "MILVUS_PORT: $MILVUS_PORT"
log "MILVUS_DATABASE: $MILVUS_DATABASE"
log "DATA_DIR: $DATA_DIR"
log "BATCH_SIZE: $BATCH_SIZE"
log "FORCE_CPU: $FORCE_CPU"
log "RUN_DATA_LOADER: $RUN_DATA_LOADER"

# Function to check if Milvus is ready
check_milvus() {
	python3 -c "
import sys
try:
    from pymilvus import connections
    connections.connect(host='$MILVUS_HOST', port='$MILVUS_PORT', user='$MILVUS_USER', password='$MILVUS_PASSWORD')
    connections.disconnect('default')
    sys.exit(0)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1
}

# Function to check if data already exists
check_existing_data() {
	python3 -c "
import sys
try:
    from pymilvus import connections, utility, db
    connections.connect(host='$MILVUS_HOST', port='$MILVUS_PORT', user='$MILVUS_USER', password='$MILVUS_PASSWORD')

    # Check if database exists
    if '$MILVUS_DATABASE' in db.list_database():
        db.using_database('$MILVUS_DATABASE')
        collections = utility.list_collections()
        if collections:
            connections.disconnect('default')
            sys.exit(0)  # Data exists

    connections.disconnect('default')
    sys.exit(1)  # No data found
except Exception:
    sys.exit(1)
" >/dev/null 2>&1
}

# Wait for Milvus to be ready (only if data loader is enabled)
if [ "$RUN_DATA_LOADER" = "true" ]; then
	log "Waiting for Milvus to be ready..."
	max_attempts=30
	attempt=1

	while [ $attempt -le $max_attempts ]; do
		if check_milvus; then
			log "Milvus is ready!"
			break
		else
			log "Milvus not ready yet (attempt $attempt/$max_attempts), waiting 10 seconds..."
			sleep 10
			attempt=$((attempt + 1))
		fi
	done

	if [ $attempt -gt $max_attempts ]; then
		log "ERROR: Milvus failed to become ready after $max_attempts attempts"
		log "Continuing without data loading..."
		export RUN_DATA_LOADER="false"
	fi
fi

# Run data loader if enabled and Milvus is ready
if [ "$RUN_DATA_LOADER" = "true" ]; then
	if check_existing_data; then
		log "Data already exists in Milvus, skipping data loading"
		echo "SKIPPED" >/tmp/data_loading_status
	else
		log "No existing data found, starting data loading process..."
		echo "IN_PROGRESS" >/tmp/data_loading_status

		# Verify data directory contents
		if [ ! -d "$DATA_DIR" ]; then
			log "ERROR: Data directory does not exist: $DATA_DIR"
			echo "FAILED" >/tmp/data_loading_status
		else
			log "Data directory contents preview:"
			find "$DATA_DIR" -name "*.parquet*" | head -5 | while read file; do
				log "  Found: $file"
			done

			# Check if data loader script exists
			if [ -f "/app/aiagents4pharma/talk2knowledgegraphs/milvus_data_dump.py" ]; then
				log "Starting Milvus data loader..."
				cd /app/aiagents4pharma/talk2knowledgegraphs

				if python3 milvus_data_dump.py; then
					log "Data loading completed successfully!"
					echo "SUCCESS" >/tmp/data_loading_status
				else
					log "ERROR: Data loading failed! Continuing with application startup..."
					echo "FAILED" >/tmp/data_loading_status
				fi
			else
				log "ERROR: Data loader script not found at /app/aiagents4pharma/talk2knowledgegraphs/milvus_data_dump.py"
				log "Continuing with application startup..."
				echo "FAILED" >/tmp/data_loading_status
			fi
		fi
	fi
else
	log "Data loader disabled"
	echo "DISABLED" >/tmp/data_loading_status
fi

# Start the main application
log "Data loading phase completed. Starting main application..."

# Ensure Python path includes the app directory
export PYTHONPATH="/app:${PYTHONPATH}"

log "Starting main application..."
log "Python path: $PYTHONPATH"
log "Note: Edge index is now loaded on-demand from Milvus (no cache file needed)"
log "Executing command: $@"
exec "$@"
