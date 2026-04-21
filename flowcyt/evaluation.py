import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.dummy import DummyClassifier

def precision_at_k(y_true, y_score, k=100):
    order = np.argsort(y_score)[::-1]
    top_k = order[:k]
    return y_true[top_k].mean()


def precision_at_min_recall(y_true, y_score, min_recall=0.05):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    valid = recall >= min_recall
    if not np.any(valid):
        return 0.0

    precision_valid = precision[valid]
    recall_valid = recall[valid]

    max_precision = precision_valid.max()

    # among those with max precision, pick the one with max recall
    best_recall = recall_valid[precision_valid == max_precision].max()

    # return a single scalar with recall as a small tie-breaker
    return max_precision + 1e-6 * best_recall

def compute_confusion(row):
    if row.y_test == 1:
        if row.y_pred == 1:
            return "True Positive"
        else:
            return "False Negative"
    else:
        if row.y_pred == 1:
            return "False Positive"
        else:
            return "True Negative"

def specificity_score(y_test, y_pred):
    return recall_score(y_test, y_pred, pos_label=0)

METRICS = (precision_score, recall_score, specificity_score, balanced_accuracy_score)
class FoldResults:
    def __init__(self, y_true, y_score, fold=-1):
        self.y_true = y_true
        self.y_score = y_score
        self.fold = fold

    def compute_threshold(
            self,
            y_true,
            y_score,
            tuning_metric=balanced_accuracy_score
        ):
        if len(y_true) == len(self.y_true):
            assert not np.all(y_true == self.y_true), "Don't tune on evaluation set!"
        threshs = [i/20 for i in range(20)]
        ba = []
        for thresh in threshs:
            y_decision = np.where(y_score > thresh, 1, 0)
            ba.append(tuning_metric(y_true, y_decision))
        idx_opt = np.argmax(ba)
        thresh_opt = threshs[idx_opt]
        return thresh_opt

    def compute_perfs(self, metrics=METRICS, thresh=0.5, **dct_kws):
        perf = []
        for metric in metrics:
            y_pred = np.where(self.y_score > thresh, 1, 0)
            perf.append({
                "fold": self.fold,
                "metric": metric.__name__,
                "score": metric(self.y_true, y_pred),
            }| dct_kws)
        return perf

    def compute_dummy_perfs(
            self,
            metrics=METRICS,
            constant=0,
            **dct_kws):
        perf = []
        clf = DummyClassifier(strategy="constant", constant=constant)
        mock_X = np.zeros(shape=(len(self.y_true), 1))
        clf.fit(mock_X, self.y_true)
        y_pred = clf.predict(mock_X)
        for metric in metrics:
            perf.append({
                "fold": self.fold,
                "metric": metric.__name__,
                "score": metric(self.y_true, y_pred),
            }| dct_kws)
        return perf

        
def shorten(name):
    kv = name.split("_")
    return kv[0]


def ppv_score(*args, **kwargs):
    return precision_score(*args, **kwargs)

def npv_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    true_neg = cm[0, 0]
    pred_neg = cm[:, 0].sum()
    if pred_neg == 0:
        return np.nan
    return true_neg / pred_neg



