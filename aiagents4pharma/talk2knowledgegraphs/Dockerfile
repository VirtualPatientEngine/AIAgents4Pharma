FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for compiling packages
RUN apt-get update && apt-get install -y \
  g++ \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy necessary files
COPY aiagents4pharma/talk2knowledgegraphs /app/aiagents4pharma/talk2knowledgegraphs
COPY docs /app/docs
COPY app /app/app
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app/frontend/streamlit_app_talk2knowledgegraphs.py", "--server.port=8501", "--server.address=0.0.0.0"]
