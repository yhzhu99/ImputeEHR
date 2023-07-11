import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from sklearn.utils.validation import check_array


class SoftImpute:
    def __init__(self, train_ds: pd.DataFrame):
        self.train_ds = train_ds
        self.require_fit = True

    def fit(self):
        # Exclude columns with time datatype
        ds = self.train_ds.copy(deep=True)
        # cols of time datatype should not be involved in KNN.
        ds = ds.iloc[:, 4:]
        ds = check_array(ds, dtype=np.float64, force_all_finite=False)
        self.imputer = SoftImpute(ds).fit()

    def execute(self, ds: pd.DataFrame):
        ds = pd.DataFrame(ds)

        imputed_ds = ds.iloc[:, 4:]
        imputed_ds = pd.DataFrame(data=self.imputer.transform(
            imputed_ds), columns=imputed_ds.columns)
        rest_ds = ds.iloc[:, :4]
        rest_ds.reset_index(drop=True, inplace=True)
        return pd.concat([rest_ds, imputed_ds], axis=1)
