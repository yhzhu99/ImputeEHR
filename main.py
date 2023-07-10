import pandas as pd

from impute_ehr.data import preprocess

df = pd.read_csv("./datasets/example.csv")
preprocess.export_patient_list_pickle(df)

impute_train_x = pd.read_pickle('./datasets/impute_train_x.pkl')
impute_val_x = pd.read_pickle('./datasets/impute_val_x.pkl')

# for ML-based imputation models, we only need the flattened list
impute_train_x_flat = preprocess.flatten_to_matrix(impute_train_x)
impute_val_x_flat = preprocess.flatten_to_matrix(impute_val_x)

