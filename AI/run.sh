#!/bin/bash

# Script to run the AI application

echo "Starting AI Chat Application..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please copy .env.example to .env and fill in your API keys."
    exit 1
fi

# Build and start services
docker-compose up --build