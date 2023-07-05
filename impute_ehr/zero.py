import pandas as pd


class ZeroImpute:
    def __init__(self, train_ds: pd.DataFrame):
        self.train_ds = train_ds
        self.require_fit = False

    def fit(self):
        pass

    def execute(self, ds: pd.DataFrame):
        return ds.fillna(0)
