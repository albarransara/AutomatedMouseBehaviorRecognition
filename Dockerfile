FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# set working directory
WORKDIR /app

# Install additional packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    wget \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install requirements
COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# Copy source code
COPY ./notebooks/ /app/src/notebooks
COPY ./data/ /app/src/data

# Open JupyterLab port
EXPOSE 8888

# Run  JupyterLab project
CMD ["jupyter","lab","--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser"]]