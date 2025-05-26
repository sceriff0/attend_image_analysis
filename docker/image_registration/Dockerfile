# Base image

FROM debian:12.5

# Install system dependencies in one step to reduce layers
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        tabix \
        libreadline-dev \
        libcairo2-dev \
        git \
        procps \
        g++ \
        python3 \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install opencv dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set up Miniconda
ENV CONDA_DIR=/opt/conda
ENV MINICONDA_HOME=~/miniconda3
ENV PATH=$CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    mkdir -p $MINICONDA_HOME && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.9.2-0-Linux-x86_64.sh -O $MINICONDA_HOME/miniconda.sh && \
    bash $MINICONDA_HOME/miniconda.sh -b -u -p $CONDA_DIR && \
    rm $MINICONDA_HOME/miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Put conda in PATH so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Bioconda packages (e.g., bftools)
RUN conda install bioconda::bftools -y

# Copy the requirements.txt file for Python dependencies
COPY requirements.txt /attend_image_analysis/requirements.txt

# Install Python dependencies via pip
RUN pip3 install -r /attend_image_analysis/requirements.txt
