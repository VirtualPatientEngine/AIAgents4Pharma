FROM python:3.12-slim

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY aiagents4pharma/talk2biomodels /app/aiagents4pharma/talk2biomodels
COPY docs /app/docs
COPY app /app/app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

# COPY T2B /app/T2B

CMD ["streamlit", "run", "app/frontend/streamlit_app_talk2biomodels.py"]