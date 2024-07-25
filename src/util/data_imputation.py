import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    """This class is used to impute nan values"""

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
        self.fill = None

    def fit(self, X):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X):
        return X.fillna(self.fill)
