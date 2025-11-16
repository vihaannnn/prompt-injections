# COMPLETE WORKFLOW GUIDE

## Project Adjustment Summary

Based on student requirements, we have adjusted the project to:

1. **Automatic Labeling Tool**: Use GPT-4 as Judge to automatically label data
2. **Data Preprocessing**: Merge two files and split into training/testing sets
3. **Complete Workflow**: From raw CSV to trained model

---

## Quick Start (3 Steps)

### Step 1: Install Additional Dependencies

```bash
pip install openai
```

### Step 2: Set OpenAI API Key

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-api-key-here"

# Or in Windows CMD
set OPENAI_API_KEY=sk-your-api-key-here
```

**Get API Key**: https://platform.openai.com/api-keys

### Step 3: Prepare Data Files

Place two CSV files in the `data/` directory:
```
data/
├── answers_extraction.csv
└── answers_extraction_without_access.csv
```

---

## Complete Workflow

### Option A: One-Click Execution (Recommended)

```cmd
complete_workflow.bat
```

This script will automatically:
1. Use GPT-4 to label all data (~1-2 hours, cost ~$0.64)
2. Merge and split training/testing sets
3. Train Judge model
4. Evaluate model performance
5. Generate charts and reports

### Option B: Step-by-Step Execution

#### Step 1: Automatic Labeling

```bash
# Label both files (batch processing)
python batch_label.py

# Or process individually
python auto_label.py --input data/answers_extraction.csv --output data/answers_extraction_labeled.csv
python auto_label.py --input data/answers_extraction_without_access.csv --output data/answers_extraction_without_access_labeled.csv
```

**Expected Output**:
- `data/answers_extraction_labeled.csv` (with new `label` column)
- `data/answers_extraction_without_access_labeled.csv` (with new `label` column)

**Time**: 1-2 hours (3,000+ samples)  
**Cost**: Approximately $0.64 USD (using gpt-4o-mini)

---

#### Step 2: Data Preprocessing

```bash
python preprocess_data.py
```

**Features**:
- Merge two labeled files
- Rename columns (`pre_prompt` → `system_prompt` etc.)
- Split training/testing sets (80/20, maintaining label balance)

**Expected Output**:
- `data/combined_labeled.csv` - Complete dataset
- `data/train.csv` - Training set
- `data/test.csv` - Testing set

---

#### Step 3: Train Judge Model

```bash
python src\judge\train.py --data data\train.csv --cross-validate --show-feature-importance
```

**Expected Output**:
- `models/judge_clf.joblib` - Trained model
- Terminal output: F1, Precision, Recall, ROC-AUC

---

#### Step 4: Evaluate Model

```bash
python src\judge\evaluate.py --data data\test.csv --plot --error-analysis --save-predictions
```

**Expected Output**:
- `results/roc_curve.png` - ROC curve
- `results/pr_curve.png` - Precision-Recall curve
- `results/error_cases.csv` - Misclassified cases
- `results/predictions.csv` - Complete prediction results

---

## Data Flow Diagram

```
Raw Data (Unlabeled)
├─ answers_extraction.csv
└─ answers_extraction_without_access.csv
        │
        ▼ [Step 1: Auto-label with GPT-4]
        │
Labeled Data
├─ answers_extraction_labeled.csv
└─ answers_extraction_without_access_labeled.csv
        │
        ▼ [Step 2: Preprocess & Merge]
        │
Training Data
├─ train.csv (80%)
└─ test.csv (20%)
        │
        ▼ [Step 3: Train Judge]
        │
Trained Model
└─ models/judge_clf.joblib
        │
        ▼ [Step 4: Evaluate]
        │
Evaluation Results
├─ results/roc_curve.png
├─ results/pr_curve.png
└─ results/error_cases.csv
```

---

## Key File Descriptions

### New Tools

| File | Function | Usage |
|------|----------|-------|
| `auto_label.py` | GPT-4 automatic labeling tool | When you have unlabeled data |
| `batch_label.py` | Batch label two files | Process multiple files simultaneously |
| `preprocess_data.py` | Data merging and splitting | After labeling is complete |
| `complete_workflow.bat` | Complete workflow | One-click full process execution |

### Documentation

| File | Content |
|------|---------|
| `AUTO_LABEL_GUIDE.md` | Detailed auto-labeling guide |
| This file | Complete workflow instructions |

---

## Cost and Time Estimation

### LLM-as-Judge Labeling Cost

| Item | Quantity | Unit Price | Subtotal |
|------|----------|------------|----------|
| File 1 (with access) | 1,651 samples | $0.0002/sample | $0.33 |
| File 2 (without access) | 1,536 samples | $0.0002/sample | $0.31 |
| **Total** | **3,187 samples** | | **$0.64** |

Approximately **20 TWD** (at 1 USD = 31 TWD exchange rate)

### Time Estimation

| Step | Time |
|------|------|
| Automatic labeling | 1-2 hours |
| Data preprocessing | < 1 minute |
| Train model | 5-10 minutes |
| Evaluate model | 2-5 minutes |
| **Total** | **Approximately 1.5-2.5 hours** |

---

## Differences from Original Design

### Original Design
- Manual data labeling required
- Time required: Manual labeling of 3,000+ samples takes several days

### Adjusted Design
- **Automatic labeling** (GPT-4 as Judge)
- Time required: 1-2 hours
- Cost: ~$0.64 USD

### Advantages
1. **Significant time savings**: From days to hours
2. **Labeling consistency**: GPT-4 applies uniform judgment criteria
3. **Reproducibility**: Same input produces same labels
4. **Scalability**: Easy to process more data

---

## Expected Results

### Labeling Statistics (Estimated)

Based on attack type analysis:

| Category | Expected Ratio | Description |
|----------|---------------|-------------|
| Attack Success (label=1) | 20-30% | Information leakage, command execution |
| Attack Failure (label=0) | 70-80% | Correct rejection |

### Model Performance (Estimated)

| Metric | Expected Value |
|--------|---------------|
| Accuracy | 0.85-0.92 |
| F1 Score | 0.83-0.90 |
| ROC-AUC | 0.90-0.95 |

---

## Troubleshooting

### Issue 1: Invalid OpenAI API Key

```bash
# Check API Key
echo $env:OPENAI_API_KEY  # PowerShell

# Test API Key
python -c "from openai import OpenAI; client = OpenAI(); print('OK')"
```

### Issue 2: Rate Limit Exceeded

```bash
# Increase delay
python auto_label.py --input data.csv --output out.csv --delay 1.0
```

### Issue 3: CSV Column Name Mismatch

Ensure your CSV contains:
- `pre_prompt`
- `attack`
- `gpt_response`

If column names are different, modify `required_cols` in `auto_label.py`.

---

## Verification of Success

When you see the following output, the workflow is complete:

```
==========================================
WORKFLOW COMPLETED SUCCESSFULLY!
==========================================

Generated files:
  Data:
    - data\combined_labeled.csv (3187 samples)
    - data\train.csv (2550 samples)
    - data\test.csv (637 samples)
  Model:
    - models\judge_clf.joblib
  Results:
    - results\roc_curve.png
    - results\pr_curve.png
```

And:
- `data/train.csv` contains `label` column (0 or 1)
- `models/judge_clf.joblib` file exists
- `results/` directory has charts

---

## Next Steps

### Analyze Results

```bash
# View evaluation metrics
Get-Content logs\evaluation.log | Select-String "Accuracy"

# View label distribution
python -c "import pandas as pd; df = pd.read_csv('data/train.csv'); print(df['label'].value_counts())"

# Analyze error cases
python -c "import pandas as pd; df = pd.read_csv('results/error_cases.csv'); print(df.head(10))"
```

### Compare Differences With/Without Access Code

```python
import pandas as pd

df1 = pd.read_csv("data/answers_extraction_labeled.csv")
df2 = pd.read_csv("data/answers_extraction_without_access_labeled.csv")

print("With Access Code:")
print(f"  Success rate: {df1['label'].mean():.1%}")

print("Without Access Code:")
print(f"  Success rate: {df2['label'].mean():.1%}")
```

This will answer the student's core question: **Impact of Access Code on attack success rate**.

---

## Getting Help

Having issues?
1. Review `AUTO_LABEL_GUIDE.md` for detailed instructions
2. Check OpenAI API account balance
3. Confirm all dependencies are installed
4. Review log files (`logs/*.log`)

---

**Good luck with your experiment!**