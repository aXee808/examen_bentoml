import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from pathlib import Path

def main(repo_path):
    # Load features and target data
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

    # save model
    joblib.dump(model, f"{repo_path}/models/elasticnet_model.pkl")
    print("Model trained and saved successfully.")
    
if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)