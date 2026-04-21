import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone



class TubePredictor(ClassifierMixin, BaseEstimator):
    def __init__(self, cell_predictor=None):
        if cell_predictor is None:
            cell_predictor = RandomForestClassifier(random_state=1234)
        self.cell_predictor = cell_predictor
    
    def sample_to_cells(self, sample, y):
        if sample.ndim != 3:
            raise ValueError(f"Sample of shape {sample.shape} has too many dims")
        n_patients, n_cells, n_markers = sample.shape
        cells = sample.reshape((n_patients * n_cells, n_markers))
        cells_targets = np.repeat(y, n_cells)
        return cells, cells_targets
    
    def fit(self, X, y):
        cells, y_cell = self.sample_to_cells(X, y)
        self.cell_predictor.fit(cells, y_cell)
        return self
    
    def predict_proba(self, X):
        tube_predictions = []
        for patient_cells in X:
            cells_probas = self.cell_predictor.predict_proba(patient_cells)
            tube_predictions.append(cells_probas.mean(axis=0))
        return np.array(tube_predictions)

class StandaloneTubePredictor(TubePredictor):
    """
    Normally tube predictor is meant for using inside the patient predictor,
    and my custom pred pipelines are meant for multi tube FCS. This just
    extracts a single tube from X to use single tube predictions in same
    pipelines.
    """
    def __init__(self, cell_predictor=None, tube_idx=0):
        self.tube_idx = tube_idx
        super().__init__(cell_predictor=cell_predictor)

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        X_tube = X[:, self.tube_idx, ...]
        return super().fit(X_tube, y)

    def predict_proba(self, X):
        X_tube = X[:, self.tube_idx, ...]
        probas = super().predict_proba(X_tube)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


class PatientPredictor(ClassifierMixin, BaseEstimator):
    """
    Patient-level classifier based on multiple cytometry tubes.

    Each tube is processed independently by a `TubePredictor` trained on
    cell-level data. Tube-level probabilities are then aggregated into a
    single patient-level probability using a pooling function.

    Parameters
    ----------
    n_tubes : int, default=3
        Number of cytometry tubes per patient.

    cell_predictor : estimator or None, default=None
        Base estimator used inside each `TubePredictor` for cell-level
        classification. If None, a RandomForestClassifier is used.

    tube_pooling_func : callable, default=np.max
        Function used to aggregate tube-level probabilities into a single
        patient-level score. Must accept an `axis` argument.
    """
    def __init__(
            self,
            n_tubes=3,
            cell_predictor=None,
            tube_pooling_func=np.max
        ):
        self.n_tubes = n_tubes
        if cell_predictor is None:
            cell_predictor = RandomForestClassifier(random_state=1234)
        self.cell_predictor = cell_predictor
        self.tube_pooling_func = tube_pooling_func

    def check_n_tubes(self, X):
        n_tubes = X.shape[1]
        if n_tubes != self.n_tubes:
            raise ValueError(f"Data has {n_tubes} tubes, expecting {self.n_tubes}")

    def fit(self, X, y):
        """
        Fit one `TubePredictor` per tube using patient labels.

        Parameters
        ----------
        X : ndarray of shape (n_patients, n_tubes, ...)
            Cytometry data. The second dimension indexes tubes.

        y : array-like of shape (n_patients,)
            Patient-level labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.classes_ = unique_labels(y)
        self.tube_predictors_ = []
        # It would be interesting to allow missing
        # tubes at inference, but for now, I don't 
        # want to silently accept it
        self.check_n_tubes(X)
        for i in range(self.n_tubes):
            tube_cells = X[:, i, ...]
            tube_predictor = TubePredictor(cell_predictor=clone(self.cell_predictor))
            tube_predictor.fit(tube_cells, y)
            self.tube_predictors_.append(tube_predictor)
        return self

    def compute_stacked_probas(self, X):
        """
        Compute per-tube positive-class probabilities for each patient.

        Parameters
        ----------
        X : ndarray of shape (n_patients, n_tubes, ...)
            Cytometry data.

        Returns
        -------
        stacked : ndarray of shape (n_tubes, n_patients)
            Positive-class probabilities for each tube and patient.
        """
        tube_probas = []
        for i in range(self.n_tubes):
            tube_cells = X[:, i, ...]
            tube_pred = self.tube_predictors_[i].predict_proba(tube_cells)
            tube_probas.append(tube_pred[:, 1])

        stacked = np.vstack(tube_probas)
        return stacked

    def predict_proba(self, X):
        """
        Predict patient-level class probabilities by pooling tube scores.

        Parameters
        ----------
        X : ndarray of shape (n_patients, n_tubes, ...)
            Cytometry data.

        Returns
        -------
        probas : ndarray of shape (n_patients, 2)
            Patient-level class probabilities following the sklearn
            convention ``[P(class=0), P(class=1)]``.
        """
        self.check_n_tubes(X)
        stacked = self.compute_stacked_probas(X)
        pooled_scores = self.tube_pooling_func(stacked, axis=0)
        return np.vstack((1-pooled_scores, pooled_scores)).T # API compatibility dark magic

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
class PatientPredictorWithClinical(PatientPredictor):
    """
    Patient-level classifier combining cytometry and tabular clinical features.

    This estimator implements a two-stage stacked model:
    1. Tube predictors are trained on cytometry data to produce tube-level
       probabilities.
    2. A clinical aggregator (logistic regression) combines out-of-fold
       tube probabilities with tabular clinical variables to make the final
       patient-level prediction.

    Out-of-fold tube predictions are used to train the aggregator in order
    to avoid data leakage.
    """
    def _oof_tube_probas(self, X, y, cv):
        """
        Generate out-of-fold tube-level probabilities for stacking.

        For each CV split, tube predictors are trained on the training fold
        and used to predict probabilities on the validation fold. This prevents
        information leakage when training a downstream aggregator.

        Parameters
        ----------
        X : ndarray of shape (n_patients, n_tubes, ...)
            Cytometry data.

        y : array-like of shape (n_patients,)
            Patient-level labels.

        cv : cross-validation splitter
            Any sklearn-compatible CV splitter yielding train/validation indices.

        Returns
        -------
        oof_probas : ndarray of shape (n_patients, n_tubes)
            Out-of-fold positive-class probabilities for each tube.
        """
        n_samples = X.shape[0]
        oof_probas = np.zeros((n_samples, self.n_tubes))

        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            for i in range(self.n_tubes):
                tube_cells_tr = X_tr[:, i, ...]
                tube_cells_val = X_val[:, i, ...]

                tube_pred = TubePredictor(
                    cell_predictor=clone(self.cell_predictor)
                )
                tube_pred.fit(tube_cells_tr, y_tr)
                oof_probas[val_idx, i] = (
                    tube_pred.predict_proba(tube_cells_val)[:, 1]
                )

        return oof_probas

    def fit(self, X, y):
        """
        Fit the stacked cytometry + clinical model.

        The training procedure is:
        1. Generate out-of-fold tube probabilities using cross-validation.
        2. Train the clinical aggregator on tabular features augmented with
           these out-of-fold tube probabilities.
        3. Retrain tube predictors on the full dataset for inference.

        Parameters
        ----------
        X : pandas.DataFrame
            Patient-level data. Must contain a "cytometry" column holding
            per-patient cytometry arrays. All other columns are treated as
            tabular clinical features.

        y : array-like of shape (n_patients,)
            Patient-level labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        cytometry = np.stack(X["cytometry"].values)
        tabular = X.drop("cytometry", axis=1).reset_index(drop=True)

        # ---------- 1. OOF tube predictions ----------
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        oof_tube_probas = self._oof_tube_probas(cytometry, y, cv)

        tube_df = pd.DataFrame(
            oof_tube_probas,
            columns=[f"tube_{i}" for i in range(self.n_tubes)]
        )

        patient_features = pd.concat((tabular, tube_df), axis=1)

        # ---------- 2. Train aggregator on OOF features ----------
        self.aggregator_ = make_pipeline(
            StandardScaler(),
            LogisticRegression()
        )
        self.aggregator_.fit(patient_features, y)

        # ---------- 3. Train final tube predictors on full data ----------
        super().fit(cytometry, y)

        self.classes_ = self.aggregator_.classes_
        self._feature_columns = patient_features.columns

        return self

    def predict_proba(self, X):
        """
        Predict patient-level class probabilities using cytometry and
        tabular clinical features.

        Tube probabilities are computed using tube predictors trained on
        the full training set, then combined with tabular features and
        passed to the trained clinical aggregator.

        Parameters
        ----------
        X : pandas.DataFrame
            Patient-level data with the same schema used during training.

        Returns
        -------
        probas : ndarray of shape (n_patients, 2)
            Patient-level class probabilities.
        """
        cytometry = np.stack(X["cytometry"].values)
        tabular = X.drop("cytometry", axis=1).reset_index(drop=True)

        tube_probas = super().compute_stacked_probas(cytometry).T

        tube_df = pd.DataFrame(
            tube_probas,
            columns=[f"tube_{i}" for i in range(self.n_tubes)]
        )

        patient_features = pd.concat((tabular, tube_df), axis=1)
        
        # Ensure that we have the same cols as the ones that were
        # seen at training time
        patient_features = patient_features[self._feature_columns]

        return self.aggregator_.predict_proba(patient_features)



