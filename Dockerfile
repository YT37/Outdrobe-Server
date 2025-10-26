# Start from a slim Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies, including the CUDA-enabled PyTorch
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user to run the application
RUN useradd -m myuser
USER myuser

# Create the cache directory and change its ownership
RUN mkdir -p /tmp/transformers_cache
RUN chown -R myuser:myuser /tmp/transformers_cache

# Expose the port the app will run on
EXPOSE 8080

# Command to run the application using Gunicorn
# We use 1 worker because this model is memory-intensive
# We set a long timeout to handle slow requests
# We use the PORT environment variable, which Cloud Run provides
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "300", "main:application"]
