FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --no-dev

# Create directories
RUN mkdir -p /app/logs /app/cache_storage /app/data

# Non-root user
RUN useradd -m -u 1000 glurpc && \
    chown -R glurpc:glurpc /app
USER glurpc

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uv", "run", "uvicorn", "glurpc.app:app", "--host", "0.0.0.0", "--port", "8000"]

