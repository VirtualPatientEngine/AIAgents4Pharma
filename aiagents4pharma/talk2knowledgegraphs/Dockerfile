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

# Install RAPIDS into the base environment
RUN conda install -y -n base -c rapidsai -c conda-forge -c nvidia \
    python=3.12 cudf=25.06 "cuda-version>=12.0,<12.9"

# Set pip default timeout and retry logic to avoid download issues
ENV PIP_DEFAULT_TIMEOUT=100
COPY requirements.txt /app
RUN pip install --upgrade pip wheel && \
    pip install --retries 5 --timeout 60 --no-cache-dir -r /app/requirements.txt

# Clean up conda & pip cache to reduce image size
RUN conda clean -afy && \
    rm -rf ~/.cache/pip

# Copy your project files into the image
COPY aiagents4pharma/talk2knowledgegraphs /app/aiagents4pharma/talk2knowledgegraphs
COPY docs /app/docs
COPY app /app/app

# Expose the port used by Streamlit
EXPOSE 8501

# Run the Streamlit app using the base environment
CMD ["streamlit", "run", "app/frontend/streamlit_app_talk2knowledgegraphs.py", "--server.port=8501", "--server.address=0.0.0.0"]