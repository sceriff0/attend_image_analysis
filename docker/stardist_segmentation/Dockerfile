#Dockerfile.stardist
FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies first (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with exact versions
RUN pip install --no-cache-dir \
    tensorflow==2.19.0 \
    stardist==0.9.1 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    cellpose==3.1.1.1 \
    scikit-image==0.25.2 \
    aicsimageio==4.14.0


# Set default command to check Python version
CMD ["python", "--version"]
