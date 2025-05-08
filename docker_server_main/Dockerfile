# Use Python 3.9 slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies, including system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean

# Copy application code and model
COPY app.py .
COPY APIKeyMiddleware.py .
COPY last.h5 .

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]