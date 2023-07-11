import numpy as np

from impute_ehr.data import preprocess


class ZeroImpute:
    def __init__(self, train_ds: list = None, val_ds: list = None):
        self.train_ds = train_ds
        self.require_fit = False
        self.require_val = False
        self.require_save_model = False
                

    def fit(self):
        pass

    def execute(self, ds: list):
        ds, lens = preprocess.flatten_to_matrix(ds)
        # fill the missing values with 0
        ds = np.nan_to_num(ds, nan=0.0)
        ds = preprocess.reverse_flatten_to_matrix(ds, lens)
        return ds
