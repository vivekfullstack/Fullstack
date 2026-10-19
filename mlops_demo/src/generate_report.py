from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Set up directory paths
DATA_DIR = Path('data')
REPORTS_DIR = Path('reports')
REPORTS_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_DIR / "processed_titanic.csv")
X = df.drop("Survived", axis=1)

X_ref = X.sample(frac=0.5, random_state=42)
X_cur = X.sample(frac=0.5, random_state=1)

# Generate report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_ref, current_data=X_cur)

# Save report
output_path = REPORTS_DIR / "monitoring_report.html"
#report.save_html(output_path)
print(f"Report generated successfully at {output_path}")

# Automatically open in default browser
webbrowser.open(output_path.resolve().as_uri())