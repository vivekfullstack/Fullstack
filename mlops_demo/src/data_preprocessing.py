from pathlib import Path
import pandas as pd

print("Starting Wine Dataset Preprocessing Pipeline... \n")

DATA_DIR = Path('data')
raw_file = DATA_DIR / 'wine.csv'
processed_file = DATA_DIR / 'processed_wine.csv'

# 1. Load Raw Data
print(f"Reading raw data from: {raw_file}...")
df = pd.read_csv(raw_file)
print(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print()

# 2. Binning Quality Scores
print("Categorizing 'quality' into ['low', 'medium', 'high']...")
bins = [0, 5, 6, 10]
labels = ["low", "medium", "high"]

df["quality_label"] = pd.cut(df["quality"], bins=bins, labels=labels)

print("Categorized Class Distribution:")
label_counts = df["quality_label"].value_counts()
for label, count in label_counts.items():
    pct = (count / len(df)) * 100
    print(f"   • {label:<8}: {count:>4} samples ({pct:.1f}%)")

# 3. Drop Original Quality Column
print("Dropping original 'quality' numeric column...")
df.drop(columns=['quality'], inplace=True)

# 4. Save Processed Data
print(f"Saving processed dataset to: {processed_file}...")
processed_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_file, index=False)

print(f"\nPreprocessing completed successfully!")
print(f"   • Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")