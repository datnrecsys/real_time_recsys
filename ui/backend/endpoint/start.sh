#!/bin/bash

# Start the Endpoint Service
echo "Starting Endpoint Service..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Start the FastAPI server
echo "Starting FastAPI endpoint server..."
uvicorn endpoint:app --host 127.0.0.1 --port 8001 --reload
