import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

class MockScaler(TransformerMixin, BaseEstimator):
    """
    Assume shape (batch_size, n_panels, n_cells, n_markers)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class CytoScaler(TransformerMixin, BaseEstimator):
    """
    Assume shape (batch_size, n_panels, n_cells, n_markers)
    """
    def __init__(self, include_tabular=False):
        self.include_tabular = include_tabular

    @staticmethod
    def _cast_to_array(X_):
        if isinstance(X_, pd.DataFrame):
            X = np.stack(X_.iloc[:, 0].to_numpy())
        else:
            X = X_
        return X

    def fit(self, X_, y=None, tabular_data=None):
        
        X = self._cast_to_array(X_)
        self.shape_ = X.shape[1:]
        target_shape = (1, X.shape[1], 1, X.shape[-1])
        self.mean_ = X.mean(axis=(0, 2)).reshape(target_shape)
        self.scale_ = X.std(axis=(0, 2)).reshape(target_shape)

        if self.include_tabular:
            if tabular_data is None:
                raise ValueError("Expected tabular metadata")
            
            if tabular_data.shape[0] != X.shape[0]:
                raise ValueError(
                    f"""
                    Cytometry and tabular arrrays have inconsistent
                    shapes. Along first dimension, X is of size
                    {X.shape[0]} and tabular data is of shape
                    {tabular_data.shape[0]}
                    """)

            self.mean_tabular_ = tabular_data.mean(axis=0)
            self.scale_tabular_ = tabular_data.std(axis=0)

        elif tabular_data is not None:
            raise ValueError(
                "Scaler was not initialized for tabular data"
            )
        return self

    def check_shape(self, X):
        if X.ndim == 3: # Single individual scaling
            X_shape = X.shape
        else:
            X_shape = X.shape[1:]

        if X_shape != self.shape_:
            raise ValueError(f"Samples have shape {X.shape[1:]}, expecting {self.shape_}")

    def transform(self, X_, y=None, tabular_data=None):

        X = self._cast_to_array(X_)
        self.check_shape(X)
        
        X_scaled = X - self.mean_
        X_scaled /= self.scale_
        X_scaled = X_scaled.reshape(X.shape)
        if self.include_tabular:
            if tabular_data is None:
                raise ValueError("Expected tabular metadata")
            tab_scaled = tabular_data - self.mean_tabular_
            tab_scaled /= self.scale_tabular_
            return X_scaled, tab_scaled
        return X_scaled

    def inverse_transform(self, X):
        check_is_fitted(self)
        X_unscaled = X * self.scale_
        X_unscaled += self.mean_
        if self.include_tabular:
            raise NotImplementedError()
        return X_unscaled



class RobustCytoScaler(TransformerMixin, BaseEstimator):
    """
    Assume shape (batch_size, n_panels, n_cells, n_markers)
    Centers with the median, scales with IQR
    """
    def __init__(self, quantile_range=(0.25, 0.75)):
        self.quantile_range = quantile_range

    def fit(self, X, y=None):
        target_shape = (1, X.shape[1], 1, X.shape[-1])
        self.median_ = np.median(X, axis=(0, 2)).reshape(target_shape)
        q_min, q_max = self.quantile_range
        _scale = np.quantile(X, q_min, axis=(0, 2)) - np.quantile(X, q_max, axis=(0, 2))
        self.scale_ = _scale.reshape(target_shape)
        return self

    def transform(self, X, y=None):
        X_scaled = X - self.median_
        X_scaled /= self.scale_
        return X_scaled

    def __repr__(self):
        return f"RobustScaler(median={self.median_}, scale={self.scale_})"


