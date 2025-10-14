# syntax=docker/dockerfile:1

# Dockerfile for the talk2aiagents4pharma application
# Multi-stage build for optimized image size with UV package manager

ARG BASE_IMAGE=ubuntu:24.04
ARG PYTHON_VERSION=3.12

FROM ${BASE_IMAGE} AS dev-base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  g++ \
  libopenblas-dev \
  libomp-dev \
  ninja-build \
  wget \
  && rm -rf /var/lib/apt/lists/*

FROM dev-base AS python-install
ARG PYTHON_VERSION=3.12

# Install Python (available in Ubuntu 24.04 default repos)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  python${PYTHON_VERSION}-venv \
  python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

FROM python-install AS uv-install
ARG INSTALL_CUDA=true
WORKDIR /app

# Install UV package manager and dependencies
COPY pyproject.toml uv.lock* ./
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
  export PATH="/root/.local/bin:$PATH" && \
  export UV_PROJECT_ENVIRONMENT="/opt/venv" && \
  uv sync --frozen --extra dev --no-install-project --python python${PYTHON_VERSION} && \
  . /opt/venv/bin/activate && \
  if [ "$INSTALL_CUDA" = "true" ]; then \
  uv pip install \
  --extra-index-url=https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  cudf-cu12 dask-cudf-cu12; \
  else \
  echo "Skipping RAPIDS packages for CPU build"; \
  fi && \
  uv cache clean

FROM ${BASE_IMAGE} AS runtime
ARG PYTHON_VERSION=3.12
ARG INSTALL_CUDA=true
LABEL maintainer="talk2aiagents4pharma"
LABEL version="1.0.0"
LABEL description="AI Agents for Pharma - Streamlit Application"

# Install runtime dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg \
  libmagic1 \
  libopenblas0 \
  libomp5 \
  python${PYTHON_VERSION} \
  && rm -rf /var/lib/apt/lists/* \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install CUDA runtime libraries required by cudf/cupy (optional)
RUN if [ "$INSTALL_CUDA" = "true" ]; then \
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-keyring.gpg && \
  echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
  > /etc/apt/sources.list.d/nvidia-cuda.list && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cuda-cudart-12-6 \
  cuda-cudart-dev-12-6 \
  cuda-nvrtc-12-6 \
  cuda-nvrtc-dev-12-6 \
  libcublas-12-6 \
  libcusparse-12-6 \
  && rm -rf /var/lib/apt/lists/*; \
  else \
  echo "Skipping CUDA installation"; \
  fi

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Copy UV virtual environment from build stage
COPY --from=uv-install /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Set working directory and create necessary directories
WORKDIR /app

# Copy application code
COPY aiagents4pharma/talk2aiagents4pharma /app/aiagents4pharma/talk2aiagents4pharma
COPY aiagents4pharma/talk2biomodels /app/aiagents4pharma/talk2biomodels
COPY aiagents4pharma/talk2knowledgegraphs /app/aiagents4pharma/talk2knowledgegraphs
COPY docs /app/docs
COPY app /app/app

# Copy and set up the entrypoint script
COPY aiagents4pharma/talk2knowledgegraphs/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Health check for production monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/health || exit 1

# Expose the default Streamlit port
EXPOSE 8501

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (can be overridden)
CMD ["streamlit", "run", "/app/app/frontend/streamlit_app_talk2aiagents4pharma.py", "--server.port=8501", "--server.address=0.0.0.0"]
