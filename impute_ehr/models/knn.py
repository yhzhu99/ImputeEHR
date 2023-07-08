import pandas as pd
from sklearn.impute import KNNImputer


class KNNImpute:
    def __init__(self, train_ds: pd.DataFrame):
        self.train_ds = train_ds
        self.require_fit = True
        self.imputer = KNNImputer(n_neighbors=5, weights="uniform")

    def fit(self):
        """ Fit the imputer on train_ds.

        Returns
        -------
        self : object
            The fitted `KNNImputer` class instance.
        """
        ds = self.train_ds.copy(deep=True)
        # cols of time datatype should not be involved in KNN.
        ds = ds.iloc[:, 4:]
        self.imputer.fit(ds)

    def execute(self, ds: pd.DataFrame):
        """ Impute all missing values in ds.

        Parameters
        ----------
        ds : DataFrame of shape (n_samples, n_features)
            The input data to complete.
            Col0 to col3 do not need to be imputed.

        Returns
        -------
        X : array-like of shape (n_samples, n_output_features)
        """
        ds = ds.copy(deep=True)

        # cols of time datatype should not be involved in KNN.
        imputed_ds = ds.iloc[:, 4:]
        imputed_ds = pd.DataFrame(
            data=self.imputer.transform(imputed_ds),
            columns=imputed_ds.columns
        )
        # value should not be negative
        imputed_ds.where(imputed_ds >= 0, 0, inplace=True)

        # get cols having datatype of time.
        # index should be reset to solve row mismatch bug
        rest_ds = ds.iloc[:, :4]
        rest_ds.reset_index(drop=True, inplace=True)

        return pd.concat([rest_ds, imputed_ds], axis=1)
