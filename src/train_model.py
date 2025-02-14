import pandas as pd
import numpy as np
import bentoml
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error,r2_score
from pathlib import Path

def main(repo_path):
    # Load train datasets
    X_train_scaled = pd.read_csv("./data/processed/X_train_scaled.csv",sep=',')
    y_train = pd.read_csv("./data/processed/y_train.csv",sep=',')
    y_train = np.ravel(y_train)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=43)

    # define model
    ratios = np.arange(0,1,0.01)
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
    
    # fit model
    model.fit(X_train_scaled,y_train)

    # load test datasets
    X_test_scaled = pd.read_csv(f"{repo_path}/data/processed/X_test_scaled.csv")
    y_test = pd.read_csv(f"{repo_path}/data/processed/y_test.csv")
    y_test = np.ravel(y_test)

    # evaluate model
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2score = r2_score(y_test, predictions)
    print(f"Evaluation du modèle : MSE = {mse} / R2 Score = {r2score}")

    # save model
    model_ref = bentoml.sklearn.save_model("admission_elasticnet", model)
    print(f"Model trained and saved successfully on bentoml : {model_ref}")
    
if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)