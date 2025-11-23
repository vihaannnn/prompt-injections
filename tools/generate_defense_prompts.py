# tools/generate_defense_prompts.py
"""
Generate Defense Prompts
Automatically generates improved system prompts for successful attacks
Uses GPT-4 to create defensive prompts based on attack patterns

AI-assisted: This module was developed with assistance from AI tools.
"""

import os
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFENSE_GENERATION_PROMPT = """You are a security expert specializing in prompt injection defense.
Given a system prompt that was successfully attacked, generate an IMPROVED system prompt that defends against this type of attack.

ORIGINAL SYSTEM PROMPT:
{system_prompt}

ATTACK THAT SUCCEEDED:
{attack}

MODEL OUTPUT (leaked information):
{output}

ACCESS CODE:
{access_code}

Your task:
1. Analyze why this attack succeeded
2. Generate a NEW, IMPROVED system prompt that:
   - Maintains the original functionality
   - Adds specific defenses against this attack type
   - Uses clear, enforceable rules
   - Is concise but comprehensive
3. Make sure the new system prompt ensures that the LLM should never disclose its' access code in any circumstances

OUTPUT FORMAT:
Return ONLY the improved system prompt text, nothing else.
Do not include explanations or metadata.
"""


def generate_defense_prompt(system_prompt, attack, output, access_code="", model="gpt-4o-mini"):
    """Generate improved system prompt using GPT-4"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": DEFENSE_GENERATION_PROMPT.format(
                        system_prompt=system_prompt,
                        attack=attack,
                        output=output,
                        access_code=access_code
                    )
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating defense: {e}")
        return None


def process_successful_attacks(input_file, output_file, model="gpt-4o-mini", delay=0.5):
    """Process all successful attacks and generate improved prompts"""
    
    print(f"Reading successful attacks from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Total successful attacks: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Map column names
    column_mapping = {
        'pre_prompt': 'system_prompt',
        'attack': 'user_prompt',
        'gpt_response': 'model_output'
    }
    
    # Check which columns exist
    system_col = 'pre_prompt' if 'pre_prompt' in df.columns else 'system_prompt'
    attack_col = 'attack' if 'attack' in df.columns else 'user_prompt'
    output_col = 'gpt_response' if 'gpt_response' in df.columns else 'model_output'
    access_code_col = 'access_code' if 'access_code' in df.columns else None
    
    print(f"\nUsing columns:")
    print(f"  System prompt: {system_col}")
    print(f"  Attack: {attack_col}")
    print(f"  Output: {output_col}")
    
    # Generate improved prompts
    improved_prompts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating defenses"):
        # Get access_code if column exists, otherwise use empty string
        access_code = row[access_code_col] if access_code_col and access_code_col in df.columns else ""
        
        improved = generate_defense_prompt(
            system_prompt=row[system_col],
            attack=row[attack_col],
            output=row[output_col],
            access_code=access_code,
            model=model
        )
        
        improved_prompts.append(improved)
        time.sleep(delay)
    
    # Add improved prompts to dataframe
    df['improved_system_prompt'] = improved_prompts
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved improved prompts to: {output_path}")
    print(f"Successfully generated: {df['improved_system_prompt'].notna().sum()} prompts")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    
    process_successful_attacks(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        delay=args.delay
    )