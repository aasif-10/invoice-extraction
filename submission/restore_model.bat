@echo off
REM Restore llama3.2-vision model for judges
REM Run this BEFORE running the extraction system

echo ============================================
echo Restoring llama3.2-vision Model
echo ============================================
echo.

REM Check if model folders exist
if not exist "%~dp0blobs" (
    echo Error: Model files (blobs/ and manifests/) not found
    echo Please ensure the submission folder contains blobs/ and manifests/ directories
    pause
    exit /b 1
)

set SOURCE_DIR=%~dp0

REM Determine Ollama models directory
REM Check custom OLLAMA_MODELS environment variable first
set OLLAMA_MODELS=%OLLAMA_MODELS%
if "%OLLAMA_MODELS%"=="" (
    set OLLAMA_MODELS=%LOCALAPPDATA%\Ollama\models
)
if not exist "%LOCALAPPDATA%\Ollama" (
    if "%OLLAMA_MODELS%"=="" (
        set OLLAMA_MODELS=%USERPROFILE%\.ollama\models
    )
)

echo Source: %SOURCE_DIR%
echo Target: %OLLAMA_MODELS%
echo.

REM Create directory if it doesn't exist
if not exist "%OLLAMA_MODELS%" mkdir "%OLLAMA_MODELS%"

echo Copying model files (this will take 1-2 minutes)...

REM Copy blobs and manifests folders to Ollama directory
xcopy "%SOURCE_DIR%blobs" "%OLLAMA_MODELS%\blobs\" /E /I /H /Y /Q
xcopy "%SOURCE_DIR%manifests" "%OLLAMA_MODELS%\manifests\" /E /I /H /Y /Q

if %errorlevel% equ 0 (
    echo.
    echo ✓ Model restored successfully!
    echo.
    echo Verifying installation...
    ollama list
    
    echo.
    echo ============================================
    echo Model ready! You can now run:
    echo   python executable.py invoice.png
    echo   streamlit run app.py
    echo ============================================
) else (
    echo.
    echo ✗ Restore failed
    pause
    exit /b 1
)

pause
