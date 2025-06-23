# Start from Python 3.11.9 base image
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    # libpq-dev is needed for psycopg2
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./

# Install Python dependencies using pip
RUN pip install --no-cache-dir \
    "apache-airflow[postgres]==2.10.2" \
    "apache-airflow-providers-docker<4.0.0,>=3.14.0" \
    "dbt-core<2.0.0,>=1.8.5" \
    "dbt-postgres<2.0.0,>=1.8.2" \
    "psycopg2<3.0.0,>=2.9.9" \
    "feast[postgres]==0.40.1"

COPY feature_pipeline/notebooks/*.ipynb ./feature_pipeline/notebooks/
COPY feature_pipeline/notebooks/*.py ./feature_pipeline/notebooks/
# For the sake of a concept introduction tutorial we would copy the dbt and feature_store folders over into Docker
# But in practice this might pose a security risk since the credentials are stored in plain text inside this Docker image
COPY feature_pipeline/dbt/ ./feature_pipeline/dbt/
COPY feature_pipeline/feature_store/ ./feature_pipeline/feature_store/
COPY src/ ./src/

WORKDIR /app/feature_pipeline/notebooks

RUN mkdir -p ./papermill-output
