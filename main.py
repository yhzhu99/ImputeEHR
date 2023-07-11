import pandas as pd

from impute_ehr import models
from impute_ehr.data import preprocess

df = pd.read_csv("./datasets/example.csv")
preprocess.export_patient_list_pickle(df)

train_x = pd.read_pickle('./datasets/impute_train_x.pkl')
val_x = pd.read_pickle('./datasets/impute_val_x.pkl')

# # for ML-based imputation models, we only need the flattened list


# methods = ["ZeroImpute", "KNNImpute", "PCAImpute", "RandomForestImpute", "MICEImpute"]
method = "ZeroImpute"
# execute imputation pipeline
# init the pipeline

model_class = getattr(models, method)
pipeline = model_class(train_x, val_x)

# fit the imputation model if required
if pipeline.require_fit:
    pipeline.fit()

ds = pipeline.execute(train_x)
# print(ds)
# save the imputation model
# TODO: save the model

