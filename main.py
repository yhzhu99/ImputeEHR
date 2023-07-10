import pandas as pd

from impute_ehr import models
from impute_ehr.data import preprocess

df = pd.read_csv("./datasets/example.csv")
preprocess.export_patient_list_pickle(df)

impute_train_x = pd.read_pickle('./datasets/impute_train_x.pkl')
impute_val_x = pd.read_pickle('./datasets/impute_val_x.pkl')

# for ML-based imputation models, we only need the flattened list
impute_train_x_flat = preprocess.flatten_to_matrix(impute_train_x)
impute_val_x_flat = preprocess.flatten_to_matrix(impute_val_x)

# print dataset size
print("Samples of training set:", len(impute_train_x_flat))
print("Samples of validation set:", len(impute_val_x_flat))

# methods = ["ZeroImpute", "KNNImpute", "PCAImpute", "RandomForestImpute", "MICEImpute"]
methods = ["ZeroImpute"]

for method in methods:
    # execute imputation pipeline
    # init the pipeline
    pipeline = getattr(models, method)(impute_train_x_flat, impute_val_x_flat)
    # fit the imputation model if required
    if pipeline.require_fit:
        pipeline.fit()
    # save the imputation model
    # TODO: save the model