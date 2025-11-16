"""
Extract Successful Attack Prompts
Extracts all successful prompt injection attacks (label=1) from labeled data
"""

import pandas as pd
from pathlib import Path

def extract_successful_attacks(input_file, output_file):
    """
    Extract successful attacks from labeled CSV
    
    Args:
        input_file: Path to labeled CSV file
        output_file: Path to output CSV file
    """
    print(f"Reading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Filter successful attacks (label = 1)
    successful = df[df['label'] == 1].copy()
    
    print(f"\nSuccessful attacks: {len(successful)}")
    print(f"Success rate: {len(successful)/len(df):.2%}")
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    successful.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"Columns: {list(successful.columns)}")
    
    return successful


if __name__ == "__main__":
    # Process both files
    files = [
        {
            "input": r"D:\LLM_judge\data\answers_extraction_labeled.csv",
            "output": r"D:\LLM_judge\results\successful_attacks_with_code.csv",
            "name": "With Access Code"
        },
        {
            "input": r"D:\LLM_judge\data\answers_extraction_without_access_labeled.csv",
            "output": r"D:\LLM_judge\results\successful_attacks_without_code.csv",
            "name": "Without Access Code"
        }
    ]
    
    for file_info in files:
        print("=" * 60)
        print(f"Processing: {file_info['name']}")
        print("=" * 60)
        
        try:
            extract_successful_attacks(file_info["input"], file_info["output"])
        except Exception as e:
            print(f"ERROR: {e}")
        
        print()
