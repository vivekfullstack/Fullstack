import webbrowser
from pathlib import Path
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

# Set up directory paths
DATA_DIR = Path('data')
REPORTS_DIR = Path('reports')
REPORTS_DIR.mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_DIR / "processed_wine.csv")

# Drop target label columns if present
target_cols = [col for col in ["quality_label", "quality"] if col in df.columns]
X = df.drop(columns=target_cols) if target_cols else df.copy()

# 2. Split into Reference (Baseline) and Current (Production) sets
X_ref = X.sample(frac=0.5, random_state=42).copy()
X_cur = X.sample(frac=0.5, random_state=1).copy()

# 3. Hardcode feature drift on specific wine chemistry columns
if 'fixed acidity' in X_cur.columns:
    X_cur['fixed acidity'] = X_cur['fixed acidity'] + 3.5  # Simulate high acidity shift

if 'volatile acidity' in X_cur.columns:
    X_cur['volatile acidity'] = X_cur['volatile acidity'] * 2.0  # Simulate vinegary batch anomaly

if 'alcohol' in X_cur.columns:
    X_cur['alcohol'] = X_cur['alcohol'] * 0.7  # Simulate lower alcohol concentration

if 'pH' in X_cur.columns:
    X_cur['pH'] = X_cur['pH'] - 0.8  # Simulate significant pH drop

# 4. Hardcode prediction drift (simulating model output distribution shift)
# Reference predictions: Normal distribution centered around high quality (1)
X_ref['prediction'] = np.random.choice([0, 1], size=len(X_ref), p=[0.2, 0.8])

# Current predictions: Shifted distribution predicting mostly low quality (0) due to feature drift
X_cur['prediction'] = np.random.choice([0, 1], size=len(X_cur), p=[0.85, 0.15])

# 5. Run Evidently Data Drift Report (evaluates features + prediction column)
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=X_ref, current_data=X_cur)

# 6. Save and open HTML report
output_path = REPORTS_DIR / "monitoring_report.html"
snapshot.save_html(str(output_path))

print(f"Report generated successfully with hardcoded drift at {output_path.resolve()}")
webbrowser.open(output_path.resolve().as_uri())