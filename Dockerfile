FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face embedding model to cache it in the Docker layer.
# This prevents startup delays or container timeouts on deployment.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')"

# Copy backend and frontend files
COPY app ./app
COPY frontend ./frontend

# Expose server port
EXPOSE 8000

# Start application server
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]
