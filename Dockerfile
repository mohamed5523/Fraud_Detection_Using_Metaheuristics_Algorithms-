FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY README.md .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Default command
CMD ["python", "src/fraud_detection/main.py", "--help"]
