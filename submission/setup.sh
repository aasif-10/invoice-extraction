#!/bin/bash
# Invoice Extraction Setup Script
# Run this once before evaluation to download the Vision LLM model

echo "Setting up Invoice Extraction System..."
echo "This will download the llama3.2-vision model (~4GB)"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed."
    echo "Please install Ollama first: https://ollama.ai"
    exit 1
fi

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 5

# Pull the model
echo "Downloading llama3.2-vision model..."
ollama pull llama3.2-vision

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Setup complete!"
    echo "You can now run: python executable.py <image.png>"
else
    echo ""
    echo "✗ Model download failed"
    exit 1
fi

# Keep Ollama server running
echo ""
echo "Ollama server is running (PID: $OLLAMA_PID)"
echo "To stop it later: kill $OLLAMA_PID"
