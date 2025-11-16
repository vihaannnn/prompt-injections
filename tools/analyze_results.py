"""
Analyze Results
Compare attack success rates and defense effectiveness
"""

import pandas as pd
from pathlib import Path

def analyze_results():
    """Analyze and compare results from both datasets"""
    
    print("=" * 60)
    print("Attack Success Rate Analysis")
    print("=" * 60)
    print()
    
    # Load labeled data
    files = {
        "with_code": "data/answers_extraction_labeled.csv",
        "without_code": "data/answers_extraction_without_access_labeled.csv"
    }
    
    results = {}
    
    for name, file_path in files.items():
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        total = len(df)
        successful = (df['label'] == 1).sum()
        failed = (df['label'] == 0).sum()
        success_rate = successful / total * 100
        
        results[name] = {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": success_rate
        }
        
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  Total attacks: {total}")
        print(f"  Successful: {successful} ({success_rate:.1f}%)")
        print(f"  Failed: {failed} ({100-success_rate:.1f}%)")
        print()
    
    # Calculate difference
    if "with_code" in results and "without_code" in results:
        diff = results["with_code"]["success_rate"] - results["without_code"]["success_rate"]
        print("=" * 60)
        print("Comparison")
        print("=" * 60)
        print(f"Success rate WITH access code: {results['with_code']['success_rate']:.1f}%")
        print(f"Success rate WITHOUT access code: {results['without_code']['success_rate']:.1f}%")
        print(f"Difference: {abs(diff):.1f} percentage points")
        
        if diff > 0:
            print(f"\nConclusion: Attacks are {abs(diff):.1f}% MORE successful with access code")
        elif diff < 0:
            print(f"\nConclusion: Attacks are {abs(diff):.1f}% LESS successful with access code")
        else:
            print("\nConclusion: No significant difference")
    
    print()
    
    # Check if improved prompts exist
    improved_files = {
        "with_code": "results/improved_prompts_with_code.csv",
        "without_code": "results/improved_prompts_without_code.csv"
    }
    
    print("=" * 60)
    print("Defense Prompt Generation")
    print("=" * 60)
    
    for name, file_path in improved_files.items():
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            generated = df['improved_system_prompt'].notna().sum()
            print(f"{name.replace('_', ' ').title()}: {generated} improved prompts generated")
        else:
            print(f"{name.replace('_', ' ').title()}: Not yet generated")
    
    print()


if __name__ == "__main__":
    analyze_results()
