FROM python:3.11-slim

WORKDIR /app

# Install uv with specific version for reproducibility
RUN pip install --no-cache-dir uv==0.5.11

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies only (no dev dependencies in production)
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run application
CMD ["sh", "-c", "uv run fastapi run --host 0.0.0.0 --port ${PORT:-8000} main.py"]
