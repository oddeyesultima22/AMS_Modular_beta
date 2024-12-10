FROM python:3.8-slim

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install h5py before other dependencies
RUN pip install h5py==3.9.0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    gcc \
    python3-dev \
    libc-dev \
    libsndfile1 \
    ffmpeg \
    libhdf5-dev \
    libblas-dev \
    cargo \
    && apt-get clean

# Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends     gcc     python3-dev     libc-dev     libsndfile1     ffmpeg     build-essential     curl     && rm -rf /var/lib/apt/lists/*
# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from the current directory to /app in the container
COPY . /app

# Create directories for input and output
# RUN mkdir -p /app/transcriptions

# Expose the port
EXPOSE 8001

CMD ["python", "AMS_Modular/main3.py"]