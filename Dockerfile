# Use official Python 3.10 slim image
FROM python:3.10-slim

# Avoid tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (for numpy/scipy etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    python3-dev \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (cache layer)
COPY requirements.txt .

# Install pip deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy rest of your app
COPY . .

# Run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

