import pandas as pd

# File paths - update these with your actual file names
input_file1 = 'extraction_robustness_dataset.csv'
input_file2 = 'hijacking_robustness_dataset.csv'
output_file1 = 'extraction_robustness_dataset_first100.csv'
output_file2 = 'hijacking_robustness_dataset_first100.csv'

# Read first CSV file and extract first 100 rows
df1 = pd.read_csv(input_file1)
df1_first100 = df1.head(100)
# Drop sample_id column if it exists
if 'sample_id' in df1_first100.columns:
    df1_first100 = df1_first100.drop('sample_id', axis=1)
df1_first100.to_csv(output_file1, index=False)
print(f"Saved first 100 rows of {input_file1} to {output_file1}")
print(f"Shape: {df1_first100.shape}")

# Read second CSV file and extract first 100 rows
df2 = pd.read_csv(input_file2)
df2_first100 = df2.head(100)
# Drop sample_id column if it exists
if 'sample_id' in df2_first100.columns:
    df2_first100 = df2_first100.drop('sample_id', axis=1)
df2_first100.to_csv(output_file2, index=False)
print(f"Saved first 100 rows of {input_file2} to {output_file2}")
print(f"Shape: {df2_first100.shape}")

print("\nProcessing complete!")