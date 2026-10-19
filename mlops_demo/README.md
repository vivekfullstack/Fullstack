

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/data_preprocessing.py
python src/model_tracking_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001
python app/main.py
