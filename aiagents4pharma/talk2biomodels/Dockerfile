# syntax=docker/dockerfile:1

# Dockerfile for the talk2biomodels application
# Multi-stage build for optimized image size

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.12
# ARG CUDF_VERSION=25.06
# ARG CUDA_VERSION_RANGE=">=12.0,<=12.8"
ARG MINIFORGE_VERSION=25.3.1-0

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

ENV PATH=/opt/conda/bin:$PATH

FROM dev-base AS conda
ARG PYTHON_VERSION=3.12
ARG MINIFORGE_VERSION=25.3.1-0
# Automatically set by buildx
ARG TARGETPLATFORM
# translating Docker's TARGETPLATFORM into miniconda arches
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MINICONDA_ARCH=aarch64  ;; \
         *)              MINICONDA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/miniconda.sh -O "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-${MINICONDA_ARCH}.sh"

# Install miniconda and base python
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build && \
    /opt/conda/bin/conda clean -ya

FROM conda AS rapids-install
ARG PYTHON_VERSION=3.12
# ARG CUDF_VERSION=25.06
# ARG CUDA_VERSION_RANGE=">=12.0,<=12.8"

WORKDIR /app

# # Install RAPIDS + cudf in the build stage
# RUN /opt/conda/bin/conda install -y -n base -c rapidsai -c conda-forge -c nvidia \
#     python=${PYTHON_VERSION} cudf=${CUDF_VERSION} "cuda-version${CUDA_VERSION_RANGE}" && \
#     /opt/conda/bin/conda clean -afy

# Copy requirements and install pip packages
COPY requirements.txt /app/
RUN /opt/conda/bin/pip install --upgrade pip wheel && \
    /opt/conda/bin/pip install --retries 15 --timeout 180 --no-cache-dir -r /app/requirements.txt && \
    rm -rf ~/.cache/pip



FROM ${BASE_IMAGE} AS runtime
ARG PYTHON_VERSION=3.12
LABEL maintainer="talk2biomodels"
LABEL version="1.0.0"
LABEL description="AI Agents for Pharma - Streamlit Application"

# Install only runtime dependencies (no build tools)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libopenblas0 \
        libomp5 \
        curl \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Copy conda environment from build stage
COPY --from=rapids-install /opt/conda /opt/conda

# Set environment variables
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Set working directory and create necessary directories
WORKDIR /app

# Copy application code
COPY aiagents4pharma/talk2biomodels /app/aiagents4pharma/talk2biomodels
COPY docs /app/docs
COPY app /app/app

# Health check for production monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose the default Streamlit port
EXPOSE 8501

# # Set the entrypoint
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (can be overridden)
CMD ["streamlit", "run", "/app/app/frontend/streamlit_app_talk2biomodels.py", "--server.port=8501", "--server.address=0.0.0.0"]