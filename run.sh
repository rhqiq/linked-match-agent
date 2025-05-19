#!/bin/bash

# Run AI Candidate Matcher
# This script runs the enhanced version of the AI Candidate Matcher

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing required packages..."
    pip install -e .
fi

# Check if the .env file exists
if [ ! -f .env ]; then
    if [ -f .env-template ]; then
        echo "No .env file found. Creating from template..."
        cp .env-template .env
        echo "Please edit the .env file to add your API key."
        echo "GEMINI_API_KEY=your_api_key_here" >> .env
        echo ""
    else
        echo "No .env or .env-template file found. Creating .env file..."
        echo "GEMINI_API_KEY=your_api_key_here" > .env
        echo "Please edit the .env file to add your API key."
        echo ""
    fi
fi

# Run the app
echo "Starting AI Candidate Matcher Pro..."
streamlit run app.py 