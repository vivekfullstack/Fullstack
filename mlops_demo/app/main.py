import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
import uvicorn

# Point to the SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def select_model():
    client = mlflow.tracking.MlflowClient()
    
    # Get the experiment created by training script
    experiment = client.get_experiment_by_name("titanic_experiment")
    if not experiment:
        print("Experiment 'titanic_experiment' not found in mlflow.db.")
        sys.exit(1)
        
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        print("No logged model runs found in experiment.")
        sys.exit(1)
        
    print("\n--- Available Models in MLFlow ---")
    for idx, run in enumerate(runs):
        run_id = run.info.run_id
        model_type = run.data.params.get("model_type", "Unknown")
        accuracy = run.data.metrics.get("accuracy", "N/A")
        print(f"[{idx}] Run ID: {run_id} | Type: {model_type} | Accuracy: {accuracy}")
        
    choice = int(input("\nSelect model index to load into app: "))
    selected_run = runs[choice]
    run_id = selected_run.info.run_id
    
    print(f"\nLoading model from Run ID: {run_id}...")
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    return loaded_model, run_id

model, selected_run_id = select_model()

app = FastAPI(title=f"Titanic Prediction API (Model: {selected_run_id[:8]})")

@app.get("/")
def root():
    return {"status": "active", "loaded_run_id": selected_run_id}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)