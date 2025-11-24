# LLM Prompt Injection Judge

Automated detection system for prompt injection attacks using hybrid rule-based and ML approaches.

## Overview

This project addresses the challenge of evaluating prompt injection attack success across thousands of test cases. Instead of manual labeling, we use GPT-4 as an automated judge to label attack outcomes, then train a lightweight hybrid classifier for fast, accurate detection.

### Research Question

Does providing an access code make prompt injection attacks more successful?

We analyze two datasets:
- `answers_extraction.csv` - attacks with access code
- `answers_extraction_without_access.csv` - attacks without access code

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Windows (batch scripts provided)

### Installation

```bash
# Clone repository
git clone https://github.com/vihaannnn/prompt-injections.git
cd prompt-injections

# Create virtual environment
python3 -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Usage

Set your OpenAI API key:
```cmd
set OPENAI_API_KEY=sk-your-key-here
set CSV_FILE_PATH=extraction_robustness_dataset_first100.csv
```

Run complete workflow:
```cmd
complete_workflow.bat
```
To run app 
```
streamlit run iterative_app.py
```

This will:
1. Auto-label both CSV files using GPT-4 (1-2 hours, ~$0.64)
2. Preprocess and split data into train/test sets
3. Train hybrid judge model
4. Evaluate performance with ROC/PR curves
5. Generate error analysis

## System Architecture

### Two-Stage Approach

**Stage 1: Auto-Labeling with GPT-4**

For each row, GPT-4 receives:
- System prompt (pre-prompt)
- User attack
- Model response

GPT-4 judges: `1` (attack successful) or `0` (attack failed)

**Stage 2: Hybrid Judge Training**

Our trained judge uses:
- **Rule-based filter** - Fast pattern matching for obvious cases (direct leaks, clear refusals)
- **ML classifier** - Logistic regression on semantic + statistical features for edge cases

## Project Structure

```
optimized_judge/
├── data/                    # CSV files (raw and labeled)
├── src/judge/               # Core modules
│   ├── features.py         # Feature extraction
│   ├── rules.py            # Rule engine
│   ├── train.py            # Model training
│   ├── evaluate.py         # Evaluation
│   └── infer.py            # Inference
├── tools/                   # Auto-labeling scripts
│   ├── auto_label.py       # file labeling
│   └── preprocess_data.py  # Data preprocessing
├── tests/                   # Testing scripts
├── models/                  # Trained models
├── results/                 # Evaluation outputs
└── logs/                    # Execution logs
└── attack_scripts/          # Scripts relating to attacking the model
```

## Performance

- **Accuracy**: 85-90%
- **Inference Speed**: 100-500 samples/second (with batching)
- **Rule Coverage**: 15+ attack patterns
- **Features**: 8 advanced detection signals

## Key Features

### Advanced Detection
- Leakage score (system prompt overlap)
- Refusal pattern matching
- Semantic similarity analysis
- Attack keyword detection

### Optimizations
- Batch processing for 10x speedup
- Feature caching
- Dynamic risk thresholds
- Pre-compiled regex patterns

## Results Analysis

After training, compare attack success rates:
```python
# Example output
With access code: 75% success rate
Without access code: 45% success rate
Difference: 30 percentage points
```

## Manual Steps (Alternative)

If you prefer step-by-step execution:

```bash
# Step 1: Label data
python tools\auto_label.py --input data\answers_extraction.csv --output data\answers_extraction_labeled.csv

# Step 2: Preprocess
python tools\preprocess_data.py

# Step 3: Train
python src\judge\train.py --data data\train.csv

# Step 4: Evaluate
python src\judge\evaluate.py --data data\test.csv
```

## Testing

Run attack suite tests:
```bash
python tests\test_attack_suite.py --model models\judge_clf.joblib
```

Run performance benchmarks:
```bash
python tests\test_performance.py --model models\judge_clf.joblib
```

## Output Files

- `models/judge_clf.joblib` - Trained classifier
- `results/roc_curve.png` - ROC curve
- `results/pr_curve.png` - Precision-Recall curve
- `results/error_cases.csv` - Misclassified samples
- `logs/training.log` - Training details
- `logs/evaluation.log` - Evaluation metrics

## Configuration

Adjust risk levels in evaluation:
```python
judge = LLMJudge(risk_level="high")  # low, medium, high
```

- **High**: Fewer false negatives (catches more attacks)
- **Low**: Fewer false positives (fewer false alarms)

## Dependencies

See `requirements.txt`:
- scikit-learn
- sentence-transformers
- openai
- pandas
- numpy
- matplotlib
- seaborn
