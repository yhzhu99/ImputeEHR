import pandas as pd

from impute_ehr import models
from impute_ehr.data import preprocess

df = pd.read_csv("./datasets/example.csv")
preprocess.export_patient_list_pickle(df)

train_x = pd.read_pickle('./datasets/impute_train_x.pkl')
val_x = pd.read_pickle('./datasets/impute_val_x.pkl')

# methods = ["ZeroImpute", "KNNImpute", "PCAImpute", "RandomForestImpute", "MICEImpute"]
method = "KNNImpute"

# execute imputation pipeline
# init the pipeline
model_class = getattr(models, method)
pipeline = model_class(train_x, val_x)

# fit the imputation model if required
if pipeline.require_fit:
    pipeline.fit()

# save the imputation model
if pipeline.require_save_model:
    pd.to_pickle(pipeline.imputer, f"./checkpoints/{method}.ckpt")

# load the model and perform imputation
model = pd.read_pickle(f"./checkpoints/{method}.ckpt")
model_class = getattr(models, method)
pipeline = model_class(model=model)
test = pipeline.execute(val_x)
