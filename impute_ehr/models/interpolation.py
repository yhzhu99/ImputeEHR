import numpy as np
from pyts.preprocessing import InterpolationImputer

from impute_ehr.data import preprocess

class InterpolationImpute:
    """Impute missing values using interpolation.

       Parameters
       ----------
       strategy : str (default = 'cubic')
           Specifies the kind of interpolation as a string
           ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
           'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
           refer to a spline interpolation of zeroth, first, second or third
           order; 'previous' and 'next' simply return the previous or next value
           of the point) or as an integer specifying the order of the spline
           interpolator to use. Default is 'linear'.
    """
    def __init__(self, train_ds: list = None, val_ds: list = None, strategy: str = "cubic"):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.require_fit = True
        self.require_val = False
        self.require_save_model = True
        self.imputer = InterpolationImputer(strategy=strategy)

    def fit(self):
        """ Fit the imputer on train_ds.
            The imputation is carried out by column

        Returns
        -------
        self : object
            The fitted `InterpolationImputer` class instance.
        """
        ds, lens = preprocess.flatten_to_matrix(self.train_ds)
        ds_t = self.transpose(ds)
        ds_t = self.fill_row_boundary(ds_t)

        self.imputer.fit_transform(ds_t)

    def execute(self, ds: list):
        """ Impute all missing values in ds.

        Parameters
        ----------
        ds : a nested list. [patient, visit, feature]

        Returns
        -------
        imputed ds : a nested list. [patient, visit, feature]
        """
        ds, lens = preprocess.flatten_to_matrix(ds)

        ds_t = self.transpose(ds)
        ds_t = self.fill_row_boundary(ds_t)
        ds_t = self.imputer.transform(ds_t)
        ds = self.transpose(ds_t)

        ds = preprocess.reverse_flatten_to_matrix(ds, lens)
        return ds

    def fill_row_boundary(self, ds: list):
        """ Impute missing values at boundary at each row of ds with row mean value.

        Parameters
        ----------
        ds : a nested list. [patient, visit, feature]

        Returns
        -------
        imputed ds : a nested list. [patient, visit, feature]
        """
        X = np.array(ds)
        for i in range(X.shape[0]):
            row = X[i]
            average = np.mean(row[~np.isnan(row)])
            if ds[i][0] == np.nan:
                ds[i][0] = average
            if ds[i][-1] == np.nan:
                ds[i][-1] = average
        return ds

    def transpose(self, ds: list):
        """ Transpose a two-dimensional list.

        Parameters
        ----------
        ds : a nested list. [patient, visit, feature]

        Returns
        -------
        ds : a nested list. [patient, visit, feature]
        """
        return [list(row) for row in zip(*ds)]

