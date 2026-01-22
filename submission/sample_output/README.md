# Sample Output Examples

This directory contains example outputs from the Document AI Pipeline.

## Files

### 1. `result.json`

Example output from processing a sample invoice. Shows the strict JSON format:

- All 6 required fields
- Document-level confidence score
- Processing time and cost estimate

### 2. `master_data_example.json`

Example master data file format showing:

- List of authorized dealers
- List of valid model names
- Horse power ranges for each model (for validation)
- Asset cost ranges for each model (for validation)

## Understanding the Output

### Field Extraction

- **dealer_name**: Normalized dealer name after fuzzy matching
- **model_name**: Exact matched model from master list
- **horse_power**: Numeric HP value (validated against expected range)
- **asset_cost**: Total cost (validated against expected range)

### Signature & Stamp

Each includes:

- `present`: Boolean indicating if detected
- `bounding_box`: [x1, y1, x2, y2] coordinates if present

### Performance Metrics

- `confidence`: Document-level accuracy score (0.0 - 1.0)
  - Target: ≥0.95 for high-quality extraction
- `processing_time_sec`: Total pipeline execution time
  - Target: ≤30 seconds
- `cost_estimate_usd`: Estimated processing cost
  - Target: <$0.01

## Interpreting Confidence Scores

| Confidence Range | Interpretation                   | Action                     |
| ---------------- | -------------------------------- | -------------------------- |
| 0.95 - 1.00      | Excellent - All fields validated | Accept                     |
| 0.85 - 0.94      | Good - Minor validation issues   | Review flagged fields      |
| 0.70 - 0.84      | Fair - Multiple issues           | Manual verification needed |
| < 0.70           | Poor - Significant errors        | Re-scan or manual entry    |

## Validation Logic

### Dealer Name (Fuzzy Match ≥90%)

- Uses RapidFuzz to find closest match in master list
- Example: "ABC Motors" → "ABC Motors Pvt Ltd" (score: 91%)

### Model Name (Exact Match Required)

- Must exactly match entry in master list
- Case-insensitive, whitespace normalized
- Example: "MF 5710" → "MF 5710" (normalized, then matched)

### Horse Power (±5% Tolerance)

- Validated against model-specific range
- Example: MF 5710 expects 70-75 HP
  - Extracted: 72.5 HP → Valid ✓
  - Extracted: 80 HP → Invalid ✗ (outside 5% tolerance)

### Asset Cost (±5% Tolerance)

- Validated against model-specific range
- Example: MF 5710 expects ₹550,000 - ₹650,000
  - Extracted: ₹575,000 → Valid ✓
  - Extracted: ₹700,000 → Invalid ✗ (outside 5% tolerance)

## Master Data Updates

To add new dealers/models, simply update `master_data.json`:

```json
{
  "dealers": ["Your New Dealer Name"],
  "models": ["NEW MODEL 123"],
  "horse_power_ranges": {
    "NEW MODEL 123": { "min": 50, "max": 55 }
  },
  "cost_ranges": {
    "NEW MODEL 123": { "min": 400000, "max": 500000 }
  }
}
```

No code changes required—the system dynamically loads master data.
