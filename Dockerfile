FROM python:3.11-slim

# Install system dependencies including ffmpeg for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY code.py .

# Create directory for temporary files
RUN mkdir -p /tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "code.py"]