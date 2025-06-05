FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "tada1:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# FROM python:3.9-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Upgrade pip
# RUN pip install --no-cache-dir --upgrade pip

# # Install packages one by one to better handle dependencies
# RUN pip install --no-cache-dir fastapi==0.109.0 \
#     && pip install --no-cache-dir uvicorn==0.27.0 \
#     && pip install --no-cache-dir pinecone-client==3.0.0 \
#     && pip install --no-cache-dir langchain-pinecone \
#     && pip install --no-cache-dir python-dotenv==1.0.0 \
#     && pip install --no-cache-dir pydantic==2.5.3 \
#     && pip install --no-cache-dir typing-extensions>=4.8.0 \
#     && pip install --no-cache-dir python-multipart==0.0.6 \
#     && pip install --no-cache-dir langchain-core==0.1.17 \
#     && pip install --no-cache-dir langchain-community==0.0.19 \
#     && pip install --no-cache-dir langchain==0.1.4 \
#     && pip install --no-cache-dir langchain-ollama==0.2.3 \
#     && pip install --no-cache-dir langchain-groq \
#     && pip install --no-cache-dir simsimd \
#     && pip install --no-cache-dir nomic \
#     && pip install --no-cache-dir tiktoken \
#     && pip install --no-cache-dir httpx>=0.25.0

# # Install NLTK and pre-download data packages to avoid runtime downloads
# RUN pip install --no-cache-dir nltk==3.8.1 \
#     && pip install --no-cache-dir rank_bm25==0.2.2 \
#     && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# # Copy application code
# COPY tada1.py .
# COPY .env .

# # Expose the port the app runs on
# EXPOSE 8000

# # Add health check to verify application availability
# HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
#   CMD curl -f http://localhost:8000/basic-health || exit 1

# # Command to run the application
# CMD ["uvicorn", "tada1:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]