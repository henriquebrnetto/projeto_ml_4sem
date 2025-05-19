from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        # If columns is None, select all columns
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            self.columns_ = self.columns
        return self

    def transform(self, X):
        return X[self.columns_]