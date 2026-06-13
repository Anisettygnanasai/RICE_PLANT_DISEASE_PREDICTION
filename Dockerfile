# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure Python output is not buffered (so Render can show crash logs immediately)
ENV PYTHONUNBUFFERED=1

# Expose the default Render port
EXPOSE 10000

# Run the FastAPI server using bash so $PORT is guaranteed to evaluate correctly
CMD ["/bin/bash", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
