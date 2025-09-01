# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit runs on 0.0.0.0 inside Docker
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
