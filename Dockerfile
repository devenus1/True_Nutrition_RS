FROM runpod/base:0.4.0-cuda11.8.0


# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., for cryptography or other libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install runpod explicitly if not in requirements 
RUN pip install --no-cache-dir runpod

# Copy source files
COPY src/ ./src/

# Expose port for API
EXPOSE 8080


# Set the command to run the handler.py which starts the Runpod serverless worker
CMD ["python", "-u", "./src/handler.py"]