# 🎯 Quick Start Guide for Judges

**Evaluation Time: ~10-15 minutes total setup + testing**

---

## Prerequisites (One-time Setup)

**Required software (if not already installed):**

1. **Ollama** (download from [ollama.com](https://ollama.com))
   - Windows: Download and run `OllamaSetup.exe` (~200 MB)
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
   - Mac: Download and install `Ollama.app`

2. **Python 3.8-3.11** (check with `python --version`)

**Note:** You do NOT need to download the llama3.2-vision model separately - it's included in the submission (~7.8 GB). The `restore_model` script will install it automatically.

---

## Step 1: Extract Submission (1 minute)

```bash
# Unzip the submission package
unzip submission.zip
cd submission/
```

**Contents:**

- `blobs/` and `manifests/` folders (~7.8 GB) - llama3.2-vision model files (unzipped)
- Python scripts and utilities
- Sample data

---

## Step 2: Install Model (1-2 minutes)

### Windows:

```cmd
restore_model.bat
```

### Linux/Mac:

```bash
bash restore_model.sh
```

**What this does:**

- Copies the llama3.2-vision model files to Ollama directory
- Starts Ollama server automatically
- Verifies model is ready

**Expected Output:**

```
✓ Model restored successfully!
NAME                      ID              SIZE
llama3.2-vision:latest    6f2f9757ae97    7.8 GB
```

---

## Step 3: Install Python Dependencies (2-3 minutes)

```bash
pip install -r requirements.txt
```

**Core dependencies:**

- requests, opencv-python, Pillow, numpy (required)
- torch, easyocr, ultralytics, spacy (optional but recommended)
- matplotlib, seaborn, pandas, streamlit (for bonus features)

---

## Step 4: Test Extraction (1 minute per invoice)

### Method 1: Command Line (Fastest)

```bash
# Test with included sample invoice
python executable.py data/sample_invoices/172863544_2_pg20.png

# Or test with your own invoice
python executable.py path/to/your/invoice.png
```

**Expected Output:**

```json
{
  "doc_id": "172863544_2_pg20",
  "fields": {
    "dealer_name": "SABAR AGROTECH",
    "model_name": "TAFE MF 241",
    "horse_power": 42.0,
    "asset_cost": 830000.0,
    "signature": {
      "present": true,
      "bbox": [850, 1400, 250, 120]
    },
    "stamp": {
      "present": true,
      "bbox": [150, 1450, 200, 150]
    }
  },
  "confidence": 0.97,
  "processing_time_sec": 28.9,
  "cost_estimate_usd": 0.002
}
```

### Method 2: Web Interface (Interactive)

```bash
streamlit run app.py
```

Then:

1. Open browser at `http://localhost:8501`
2. Upload invoice PNG
3. Click "Extract Fields"
4. View results and download JSON

---

## Step 5: Verify Bonus Features (Optional, 5 minutes)

### EDA Analysis

```bash
# Open Jupyter notebook
jupyter notebook IntelliExtract_EDA_Analysis.ipynb

# Or view pre-exported HTML
open IntelliExtract_EDA_Analysis.html
```

**What to see:**

- 9 comprehensive analysis sections
- 12+ visualizations (state distribution, language analysis, processing time, confidence metrics)
- Production-ready insights

### Error Analysis

```bash
python error_analysis.py sample_output/result.json
```

**Outputs to `error_analysis_output/`:**

- Error category distribution charts
- Severity analysis
- Failure case reports (JSON + PNG)

---

## Performance Expectations

| Metric               | Expected Value | Notes                                |
| -------------------- | -------------- | ------------------------------------ |
| **Processing Time**  | <30 seconds    | Per invoice, 95th percentile         |
| **Confidence**       | >90%           | For clear/typed invoices             |
| **Accuracy**         | 95%+           | Field extraction success rate        |
| **Fields Extracted** | 6/6            | All fields including signature/stamp |

---

## Troubleshooting

### Issue: "Ollama not found"

**Solution:**

```bash
# Install Ollama
# Windows: Download from https://ollama.ai
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Mac: brew install ollama

# Then re-run restore_model.bat/sh
```

### Issue: "Model not loaded"

**Solution:**

```bash
# Manually start Ollama
ollama serve &

# Check if model exists
ollama list

# If not listed, re-run restore script
restore_model.bat
```

### Issue: "Python dependencies failed"

**Solution:**

```bash
# Install core dependencies only (minimal)
pip install requests opencv-python Pillow numpy

# The system will work with just these (OCR/YOLO optional)
```

### Issue: "Processing too slow (>30s)"

**Solution:**

- First run is always slower (~40s) - caches load
- Subsequent runs: <30s consistently
- Check Ollama server is running: `ollama list`

### Issue: "Low confidence scores"

**Solution:**

- Check invoice quality (resolution, clarity)
- System is lenient - even poor quality invoices get extracted
- Confidence <80% still provides results with warnings

---

## Evaluation Checklist

- [ ] Model restored successfully (`ollama list` shows llama3.2-vision)
- [ ] Dependencies installed (`pip list | grep opencv`)
- [ ] Sample invoice extracted successfully
- [ ] Output JSON matches required schema
- [ ] Processing time <30 seconds
- [ ] Confidence score >80%
- [ ] All 6 fields extracted (dealer, model, HP, cost, signature, stamp)
- [ ] Bonus: EDA notebook runs/HTML viewable
- [ ] Bonus: Streamlit web app works
- [ ] Bonus: Error analysis generates reports

---

## Quick Validation Commands

```bash
# 1. Check Ollama model
ollama list | grep llama3.2-vision

# 2. Check Python packages
pip list | grep -E "opencv|requests|Pillow"

# 3. Run extraction
python executable.py data/sample_invoices/172863544_2_pg20.png

# 4. Validate output format
cat sample_output/result.json | python -m json.tool

# 5. Check processing time (should be <30s)
time python executable.py data/sample_invoices/172863544_2_pg20.png
```

---

## Support

If you encounter issues during evaluation:

1. **Check logs:** `ollama.log`, `app.log` (if present)
2. **Verify setup:** All steps in this guide completed
3. **Test system:** Run sample invoice first before custom invoices
4. **Review README:** Detailed troubleshooting in main README.md

---

**Expected Total Time:**

- Setup (Steps 1-3): ~5-7 minutes
- Testing (Step 4): ~2-3 minutes per invoice
- Bonus Review (Step 5): ~5 minutes
- **Total: 15-20 minutes for complete evaluation**

🎯 **System is ready when:** `ollama list` shows llama3.2-vision AND sample invoice extracts successfully
