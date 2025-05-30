# # Multi-stage build for optimized production image
# FROM python:3.9-slim as builder

# # Set working directory
# WORKDIR /app

# # Install system dependencies for building
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Upgrade pip and install wheel for faster builds
# RUN pip install --no-cache-dir --upgrade pip wheel

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Pre-download NLTK data to avoid runtime downloads
# RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/share/nltk_data')"

# # Production stage
# FROM python:3.9-slim as production

# # Set working directory
# WORKDIR /app

# # Create non-root user for security
# RUN groupadd -r appuser && useradd -r -g appuser appuser

# # Install only runtime dependencies
# RUN apt-get update && apt-get install -y \
#     curl \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# # Copy Python packages from builder stage
# COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# COPY --from=builder /usr/local/bin /usr/local/bin

# # Copy NLTK data from builder stage
# COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data

# # Set NLTK data path
# ENV NLTK_DATA=/usr/share/nltk_data

# # Copy application code
# COPY tada1.py .
# COPY .env .

# # Set environment variables for optimization
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV USE_NUMPY=1

# # Change ownership to non-root user
# RUN chown -R appuser:appuser /app

# # Switch to non-root user
# USER appuser

# # Expose the port
# EXPOSE 8000

# # Add comprehensive health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
#   CMD curl -f http://localhost:8000/basic-health || exit 1

# # Command to run the application with optimized settings
# CMD ["uvicorn", "tada1:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info", "--access-log"]
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
    && pip install --no-cache-dir langchain-ollama==0.2.3 \
    && pip install --no-cache-dir langchain-groq \
    && pip install --no-cache-dir simsimd \
    && pip install --no-cache-dir nomic \
    && pip install --no-cache-dir tiktoken \
    && pip install --no-cache-dir httpx>=0.25.0

# Install NLTK and pre-download data packages to avoid runtime downloads
RUN pip install --no-cache-dir nltk==3.8.1 \
    && pip install --no-cache-dir rank_bm25==0.2.2 \
    && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY tada1.py .
COPY .env .

# Expose the port the app runs on
EXPOSE 8000

# Add health check to verify application availability
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/basic-health || exit 1

# Command to run the application
CMD ["uvicorn", "tada1:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]