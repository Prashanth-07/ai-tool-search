FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install packages one by one to better handle dependencies
RUN pip install --no-cache-dir fastapi==0.109.0 \
    && pip install --no-cache-dir uvicorn==0.27.0 \
    && pip install --no-cache-dir pinecone-client==3.0.0 \
    && pip install --no-cache-dir python-dotenv==1.0.0 \
    && pip install --no-cache-dir pydantic==2.5.3 \
    && pip install --no-cache-dir typing-extensions>=4.8.0 \
    && pip install --no-cache-dir python-multipart==0.0.6 \
    && pip install --no-cache-dir langchain-core==0.1.17 \
    && pip install --no-cache-dir langchain-community==0.0.19 \
    && pip install --no-cache-dir langchain==0.1.4 \
    && pip install --no-cache-dir langchain-ollama==0.2.3

# Copy application code
COPY tada.py .
COPY .env .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "tada:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]