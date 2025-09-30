FROM runpod/base:0.4.0-cuda11.8.0


# Set working directory
WORKDIR /app

# Copy source files
COPY runpod/src/ ./src/

# Copy model artifacts
COPY runpod/models/ ./models/

# Create symlink for python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install system dependencies if needed (e.g., for cryptography or other libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY runpod/src/requirements.txt .

# Install Python dependencies
# Upgrade pip and install dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r runpod/src/requirements.txt --no-cache-dir


# Install runpod explicitly if not in requirements 
RUN pip install --no-cache-dir runpod


# Expose port for API
EXPOSE 8080


# Set the command to run the handler.py which starts the Runpod serverless worker
CMD ["python", "-u", "./src/handler.py"]