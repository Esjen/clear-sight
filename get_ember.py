import pandas as pd
import json
import os
import glob

DATA_DIR = "data/ember_2018"

def load_jsonl_sample(filepath, n_rows=10000):
    """Load first n_rows from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Check what files exist
jsonl_files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
print(f"Found {len(jsonl_files)} JSONL files:")

for f in jsonl_files:
    print(f"  - {os.path.basename(f)}")

if not jsonl_files:
    print("\nERROR: No JSONL files found!")
    print("The ember package did not download the dataset.")
    print("Please download manually from: https://github.com/elastic/ember/releases")
    exit(1)

# Load a sample from the first training file
print(f"\nLoading sample from {os.path.basename(jsonl_files[0])}...")
df = load_jsonl_sample(jsonl_files[0], n_rows=10000)

print(f"\nShape: {df.shape}")
print(f"\nColumns (first 10): {list(df.columns[:10])}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

print("\n" + "="*50)
print("SUCCESS! EMBER data is accessible.")
print("You can now run your EDA notebook.")
print("="*50)