import numpy as np
from sklearn.preprocessing import StandardScaler

from pypots.data import load_specific_dataset, masked_fill, mcar
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae

# Data preprocessing. Tedious, but PyPOTS can help. 🤓
data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
X = data['X']
num_samples = len(X['RecordID'].unique())
X = X.drop(['RecordID', 'Time'], axis = 1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
X = masked_fill(X, 1 - missing_mask, np.nan)
dataset = {"X": X}
print(dataset["X"].shape)  # (11988, 48, 37), 11988 samples, 48 time steps, 37 features

import pdb

# pdb.set_trace()

# initialize the model
saits = SAITS(
    n_steps=48,
    n_features=37,
    n_layers=2,
    d_model=256,
    d_inner=128,
    n_heads=4,
    d_k=64,
    d_v=64,
    dropout=0.1,
    epochs=10,
    saving_path="examples/saits", # set the path for saving tensorboard logging file and model checkpoint
    model_saving_strategy="best", # only save the model with the best validation performance
)

# # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
# saits.fit(dataset)
# # impute the originally-missing values and artificially-missing values
# imputation = saits.impute(dataset)
# # calculate mean absolute error on the ground truth (artificially-missing values)
# mae = cal_mae(imputation, X_intact, indicating_mask)

# # the best model has been already saved, but you can still manually save it with function save_model() as below
# saits.save_model(saving_dir="examples/saits",file_name="manually_saved_saits_model")
# # you can load the saved model into a new initialized model
saits.load_model("examples/saits/manually_saved_saits_model.pypots")
