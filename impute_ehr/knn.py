import pandas as pd
import pandas as pd
from sklearn.impute import KNNImputer
class KNNImpute:
    def __init__(self, train_ds: pd.DataFrame):
        self.train_ds = train_ds
        self.require_fit = False

    def fit(self):
        pass

    def execute(self, ds: pd.DataFrame):
        ds = ds.copy(deep=True)

        # cols of time datatype should not be involved in KNN.
        imputed_ds = ds.iloc[:, 4:]
        imputed_ds = pd.DataFrame(
            data=KNNImputer(n_neighbors=5, weights="uniform").fit_transform(imputed_ds),
            columns=imputed_ds.columns
        )

        # get cols having datatype of time.
        # index should be reset to solve row mismatch bug
        res_ds = ds.iloc[:, :4]
        res_ds.reset_index(drop=True, inplace=True)

        return pd.concat([res_ds, imputed_ds], axis=1)

