import pandas as pd
from sklearn.model_selection import train_test_split
import impute_ehr

# divide dataset
train_size, test_size, val_size = 0.7, 0.1, 0.2
ds = pd.read_csv("./datasets/example.csv")
train_ds, tmp_ds = train_test_split(ds, train_size=train_size, random_state=17900)
test_ds, val_ds = train_test_split(tmp_ds, train_size=test_size / (test_size + val_size), random_state=1190)

print("训练集大小：", len(train_ds))
print("测试集大小：", len(test_ds))
print("验证集大小：", len(val_ds))

# execute imputation pipeline
# fill 0
# no need to divide dataset. just fill the original dataset
fill0_impute = impute_ehr.Fill0Impute(ds)
fill0_result_ds = fill0_impute.execute()
fill0_result_ds.to_csv('./output/example_fill0_result.csv')

# PCA
pca_impute = impute_ehr.PCAImpute(train_ds, test_ds)
train_ds = pca_impute.execute()
