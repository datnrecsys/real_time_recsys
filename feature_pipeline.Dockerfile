# Start from Python 3.11.9 base image
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    # libpq-dev is needed for psycopg2
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Create and set the working directory
WORKDIR /app

# Copy Poetry files
COPY poetry.lock pyproject.toml ./

# Install Python dependencies using Poetry
RUN poetry install --only feature-pipeline

COPY feature_pipeline/notebooks/*.ipynb ./feature_pipeline/notebooks/
COPY feature_pipeline/notebooks/*.py ./feature_pipeline/notebooks/
# For the sake of a concept introduction tutorial we would copy the dbt and feature_store folders over into Docker
# But in practice this might pose a security risk since the credentials are stored in plain text inside this Docker image
COPY feature_pipeline/dbt/ ./feature_pipeline/dbt/
COPY feature_pipeline/feature_store/ ./feature_pipeline/feature_store/
COPY src/ ./src/

WORKDIR /app/feature_pipeline/notebooks

RUN mkdir -p ./papermill-output

ENTRYPOINT ["poetry", "run"]