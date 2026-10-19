from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Use SQLite backend in project root
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine_experiment")

DATA_DIR = Path('data')
df = pd.read_csv(DATA_DIR / 'processed_wine.csv')

X = df.drop('quality_label', axis=1)
y = df['quality_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to train and evaluate
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# Define trusted types for skops tree serialization
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Train, evaluate, and log each model in its own MLflow run
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Pass skops_trusted_types and use name instead of deprecated artifact_path
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES
        )

        print(f"[{name}] Logged run {run.info.run_id} with Accuracy: {acc:.4f}")