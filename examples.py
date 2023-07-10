from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from impute_ehr import models

# set random seed
random_seed = 42

# split dataset into train, test, val set
train_size, test_size, val_size = 0.7, 0.1, 0.2
ds = pd.read_csv("./datasets/example.csv")
train_ds, val_test_ds = train_test_split(
    ds, train_size=train_size, random_state=random_seed)
val_ds, test_ds = train_test_split(
    val_test_ds, train_size=val_size/(val_size+test_size), random_state=random_seed)

# print dataset size
print("Samples of training set:", len(train_ds))
print("Samples of validation set:", len(val_ds))
print("Samples of test set:", len(test_ds))

# methods = ["ZeroImpute", "KNNImpute", "PCAImpute", "RandomForestImpute", "MICEImpute"]
methods = ["ZeroImpute"]

for method in methods:
    # execute imputation pipeline
    # init the pipeline
    pipeline = getattr(models, method)(train_ds)
    # fit the imputation model if required
    if pipeline.require_fit:
        pipeline.fit()
    # execute the imputation pipeline
    train_ds_imputed = pipeline.execute(train_ds)
    val_ds_imputed = pipeline.execute(val_ds)
    test_ds_imputed = pipeline.execute(test_ds)
    # save the imputed dataset
    Path(f"./datasets/imputed/{method}").mkdir(parents=True, exist_ok=True)
    train_ds_imputed.to_csv(f"./datasets/imputed/{method}/train.csv", index=False)
    val_ds_imputed.to_csv(f"./datasets/imputed/{method}/val.csv", index=False)
    test_ds_imputed.to_csv(f"./datasets/imputed/{method}/test.csv", index=False)
