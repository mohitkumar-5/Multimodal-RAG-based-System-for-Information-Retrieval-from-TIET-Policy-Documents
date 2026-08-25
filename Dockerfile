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


# Copy application files, PDFs, and ingestion script
COPY app ./app
COPY frontend ./frontend
COPY output ./output
COPY ingest.py .

# Build local vector database index during Docker build
RUN python ingest.py

# Expose server port
EXPOSE 8000

# Start application server
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]
