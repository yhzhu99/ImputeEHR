import pandas as pd
import impute_ehr

ds = pd.read_csv("./datasets/example.csv")
# train_ds =
# val_ds = 

# execute imputation pipeline
pca_impute = impute_ehr.PCAImpute(ds)
train_ds = pca_impute.execute()
