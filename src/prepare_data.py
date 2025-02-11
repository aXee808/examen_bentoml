import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
from sklearn.preprocessing import StandardScaler
import os

def main(root_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath_candidats = f"{root_path}/data/raw/admission.csv"
        
    process_data(root_path, input_filepath_candidats)

def process_data(root_path, input_filepath_candidats):
    # Import datasets
    df = import_dataset(input_filepath_candidats, sep=",")

   # Drop columns
    df = drop_columns(df)

    # Replace features names
    df = replace_features_names(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Create folder if necessary
    create_folder_if_necessary(f"{root_path}/data/processed")

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, f"{root_path}/data/processed")

    # Normalize X_train & X_test, then save them to output file paths
    normalize_data(X_train,X_test,f"{root_path}/data/processed")

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def replace_features_names(df):
    # Replace columns names
    dico={"GRE Score":"gre_score","TOEFL Score":"toefl_score","University Rating":"university_rating","SOP":"sop_rating","LOR ":"lor_rating","CGPA":"cgpa_rating","Research":"research_experience","Chance of Admit ":"admission_proba"}
    df=df.rename(dico,axis=1)
    return df

def drop_columns(df):
    # Drop columns
    df.drop(['Serial No.'], axis=1, inplace=True)
    return df

def split_data(df):
    # Split data into training and testing sets
    target = df['admission_proba']
    feats = df.drop(['admission_proba'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

def normalize_data(X_train, X_test, output_folderpath):
    # Normalise
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save dataframes to their respective output file paths
    save_dataframes_normalized(X_train_scaled, X_test_scaled, output_folderpath)

def save_dataframes_normalized(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            pd.DataFrame(file).to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

     # get root folder path (local github repo root path)
    repo_path = Path(__file__).parent.parent
    main(repo_path)