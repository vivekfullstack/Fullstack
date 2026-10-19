from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
df = pd.read_csv(DATA_DIR / 'wine.csv')

# Binning
bins = [0, 5, 6, 10]
labels = ["low", "medium", "high"]

df["quality_label"] = pd.cut(df["quality"], bins=bins, labels=labels)

# Drop unused columns
df.drop(columns=['quality'], inplace=True)

# Save processed data
df.to_csv(DATA_DIR / 'processed_wine.csv', index=False)
