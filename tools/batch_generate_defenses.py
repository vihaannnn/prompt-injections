"""
Batch Generate Defense Prompts
Processes both successful attack files and generates improved prompts
"""

import subprocess
import sys
from pathlib import Path

def run_defense_generation(input_file, output_file, description):
    """Run defense generation for a single file"""
    print("=" * 60)
    print(f"Processing: {description}")
    print("=" * 60)
    
    if not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        return False
    
    cmd = [
        sys.executable,
        "tools/generate_defense_prompts.py",
        "--input", input_file,
        "--output", output_file,
        "--model", "gpt-4o-mini",
        "--delay", "0.5"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Completed: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error: {e}")
        return False


if __name__ == "__main__":
    files = [
        {
            "input": "results/successful_attacks_with_code.csv",
            "output": "results/improved_prompts_with_code.csv",
            "description": "Successful Attacks (With Access Code)"
        },
        {
            "input": "results/successful_attacks_without_code.csv",
            "output": "results/improved_prompts_without_code.csv",
            "description": "Successful Attacks (Without Access Code)"
        }
    ]
    
    print("=" * 60)
    print("Defense Prompt Generation - Batch Processing")
    print("=" * 60)
    print()
    
    results = []
    for file_info in files:
        success = run_defense_generation(
            file_info["input"],
            file_info["output"],
            file_info["description"]
        )
        results.append(success)
        print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total: {len(files)}")
    print(f"Successful: {sum(results)}")
    print(f"Failed: {len(files) - sum(results)}")
