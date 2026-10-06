# SatQuery AI - Verification-First Multimodal EO Assistant
# Production Container Image (Python 3.11 Profile D Core)
FROM python:3.11-slim

WORKDIR /app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with CPU PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application assets
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY README.md .

# Install local package
RUN pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
