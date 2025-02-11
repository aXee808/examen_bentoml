import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error,r2_score

def main(repo_path):
    # load X_train_scaled,X_test_scaled, y_train,y_test
    X_test_scaled = pd.read_csv(f"{repo_path}/data/processed/X_test_scaled.csv")
    y_test = pd.read_csv(f"{repo_path}/data/processed/y_test.csv")
    y_test = np.ravel(y_test)

    model = load(f"{repo_path}/models/elasticnet_model.pkl")
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2score = r2_score(y_test, predictions)
    print(f"Evaluation du modèle : MSE = {mse} / R2 Score = {r2score}")
if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)