FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY sdr/ sdr/
COPY config/ config/

# Create data directory for SQLite and logs
RUN mkdir -p data/logs

CMD ["python", "-m", "sdr.main"]
