"""
Data preprocessing and merging script.

This script converts the labeled datasets into the training format expected by the LLM Judge:
- Merge two labeled CSV files (with access code / without access code)
- Normalize column names
- Split into train/test sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def preprocess_and_merge(
    file1: str,
    file2: str,
    output_combined: str,
    output_train: str,
    output_test: str,
    test_ratio: float = 0.2
):
    """
    Merge two labeled CSV files and split into train/test sets.
    
    Args:
        file1: First labeled file (with access code).
        file2: Second labeled file (without access code).
        output_combined: Output path for the merged full dataset.
        output_train: Output path for the training set.
        output_test: Output path for the test set.
        test_ratio: Test set ratio (default 0.2).
    """
    print("="*60)
    print("Data Preprocessing & Merging")
    print("="*60)
    
    # Load data
    print(f"\nLoading {file1}...")
    df1 = pd.read_csv(file1)
    print(f"  Loaded {len(df1)} samples")
    
    print(f"\nLoading {file2}...")
    df2 = pd.read_csv(file2)
    print(f"  Loaded {len(df2)} samples")
    
    # Merge
    print("\nMerging datasets...")
    combined = pd.concat([df1, df2], ignore_index=True)
    print(f"  Total samples: {len(combined)}")
    
    # Rename columns to match training script expectations
    print("\nRenaming columns...")
    combined = combined.rename(columns={
        'pre_prompt': 'system_prompt',
        'attack': 'user_prompt',
        'gpt_response': 'model_output'
    })
    
    # Keep only the required columns
    required_cols = ['system_prompt', 'user_prompt', 'model_output', 'label']
    combined = combined[required_cols]
    
    # Check for missing values
    if combined.isnull().any().any():
        print("\nWarning: Found missing values, dropping rows...")
        before = len(combined)
        combined = combined.dropna()
        after = len(combined)
        print(f"  Dropped {before - after} rows")
    
    # Dataset summary
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    label_counts = combined['label'].value_counts()
    total = len(combined)
    
    print(f"Total samples: {total}")
    print(f"  Successful attacks (label=1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%)")
    print(f"  Failed attacks (label=0):     {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)")
    
    # Save combined dataset
    print(f"\nSaving combined dataset to {output_combined}...")
    combined.to_csv(output_combined, index=False)
    
    # Train/test split (stratified by label)
    print(f"\nSplitting into train/test ({int((1-test_ratio)*100)}% / {int(test_ratio*100)}%)...")
    
    train, test = train_test_split(
        combined,
        test_size=test_ratio,
        stratify=combined['label'],  # preserve label distribution
        random_state=42
    )
    
    print(f"  Training set: {len(train)} samples")
    train_label_counts = train['label'].value_counts()
    print(f"    label=1: {train_label_counts.get(1, 0)} ({train_label_counts.get(1, 0)/len(train)*100:.1f}%)")
    print(f"    label=0: {train_label_counts.get(0, 0)} ({train_label_counts.get(0, 0)/len(train)*100:.1f}%)")
    
    print(f"  Test set: {len(test)} samples")
    test_label_counts = test['label'].value_counts()
    print(f"    label=1: {test_label_counts.get(1, 0)} ({test_label_counts.get(1, 0)/len(test)*100:.1f}%)")
    print(f"    label=0: {test_label_counts.get(0, 0)} ({test_label_counts.get(0, 0)/len(test)*100:.1f}%)")
    
    # Save train/test sets
    print(f"\nSaving training set to {output_train}...")
    train.to_csv(output_train, index=False)
    
    print(f"Saving test set to {output_test}...")
    test.to_csv(output_test, index=False)
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. {output_combined} - Full dataset ({len(combined)} samples)")
    print(f"  2. {output_train} - Training set ({len(train)} samples)")
    print(f"  3. {output_test} - Test set ({len(test)} samples)")
    print(f"\nNext steps:")
    print(f"  python src/judge/train.py --data {output_train}")
    print(f"  python src/judge/evaluate.py --data {output_test}")
    
    return combined, train, test


def main():
    parser = argparse.ArgumentParser(description="Preprocess and merge labeled datasets")
    parser.add_argument("--file1", type=str, default="data/answers_extraction_labeled.csv")
    parser.add_argument("--file2", type=str, default="data/answers_extraction_without_access_labeled.csv")
    parser.add_argument("--output-combined", type=str, default="data/combined_labeled.csv")
    parser.add_argument("--output-train", type=str, default="data/train.csv")
    parser.add_argument("--output-test", type=str, default="data/test.csv")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    
    args = parser.parse_args()
    
    preprocess_and_merge(
        file1=args.file1,
        file2=args.file2,
        output_combined=args.output_combined,
        output_train=args.output_train,
        output_test=args.output_test,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()