# Dockerfile for the talk2aiagents4pharma application
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies required for compiling Python packages
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install RAPIDS + cudf
RUN conda install -y -n base -c rapidsai -c conda-forge -c nvidia \
    python=3.12 cudf=25.06 "cuda-version>=12.0,<=12.8"

# Install pip packages
COPY requirements.txt /app
RUN pip install --upgrade pip wheel && \
    pip install --retries 15 --timeout 180 --no-cache-dir -r /app/requirements.txt && \
    conda clean -afy && rm -rf ~/.cache/pip

# Copy app code
COPY aiagents4pharma/talk2aiagents4pharma /app/aiagents4pharma/talk2aiagents4pharma
COPY aiagents4pharma/talk2biomodels /app/aiagents4pharma/talk2biomodels
COPY aiagents4pharma/talk2knowledgegraphs /app/aiagents4pharma/talk2knowledgegraphs
COPY docs /app/docs
COPY app /app/app

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app/frontend/streamlit_app_talk2aiagents4pharma.py", "--server.port=8501", "--server.address=0.0.0.0"]