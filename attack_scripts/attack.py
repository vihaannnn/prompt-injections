import csv
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def process_csv(input_file='extraction_robustness_dataset_first100.csv', output_file='answers_extraction_without_answer.csv'):
    """
    Read CSV file, process each row with OpenAI API, and save results.
    Writes to output file one row at a time to avoid storing all results in memory.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    
    # Read the input CSV to get fieldnames and count rows
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['gpt_response']
        rows = list(reader)
    
    total_rows = len(rows)
    print(f"Processing {total_rows} rows...")
    
    # Open output file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each row
        for idx, row in enumerate(rows, 1):
            access_code = row['access_code']
            attack = row['attack']
            pre_prompt = row['improved_system_prompt']
            
            # Combine access_code and attack for the user message
            user_message = f"Access Code - {access_code}, {attack}"
            
            print(f"Processing row {idx}/{total_rows}...")
            
            try:
                # Make API call to OpenAI
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": pre_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Extract the response
                ai_response = response.choices[0].message.content
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                ai_response = f"ERROR: {str(e)}"
            
            # Add the response to the row data
            row['gpt_response'] = ai_response
            
            # Write this row immediately to the output file
            writer.writerow(row)
            out_f.flush()  # Ensure it's written to disk immediately
            
            print(f"Row {idx} saved to {output_file}")
    
    print(f"\nProcessing complete! All results saved to {output_file}")

if __name__ == "__main__":
    # Set your input filename here
    INPUT_FILE = 'improved_prompts_without_code.csv'  # Change this to your CSV filename
    OUTPUT_FILE = 'improved_answers_extraction_without_code.csv'
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found!")
        print("Please make sure your CSV file is in the same directory as this script.")
    else:
        process_csv(INPUT_FILE, OUTPUT_FILE)