@echo off
REM Invoice Extraction Setup Script for Windows
REM Run this once before evaluation to download the Vision LLM model

echo Setting up Invoice Extraction System...
echo This will download the llama3.2-vision model (~4GB)
echo.

REM Check if Ollama is installed
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Ollama is not installed.
    echo Please install Ollama first: https://ollama.ai
    exit /b 1
)

REM Start Ollama server in background
echo Starting Ollama server...
start /B ollama serve
timeout /t 5 /nobreak >nul

REM Pull the model
echo Downloading llama3.2-vision model...
ollama pull llama3.2-vision

if %errorlevel% equ 0 (
    echo.
    echo Setup complete!
    echo You can now run: python executable.py ^<image.png^>
) else (
    echo.
    echo Model download failed
    exit /b 1
)

echo.
echo Ollama server is running in background
