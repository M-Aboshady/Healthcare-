# Use official slim image with Python 3.10
FROM python:3.10-slim

# Avoid interactive tzdata setup
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps for scipy/numpy etc.
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

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy rest of app
COPY . .

# Run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

