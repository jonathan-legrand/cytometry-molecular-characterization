import copy
import joblib
import numpy as np
import pandas as pd
import torch

def read_fold_scores(exppath, n_folds=10):
    y_test = []
    y_pred = []
    for i in range(n_folds):
        y_test.append(joblib.load(exppath / f"fold-{i}_true"))
        y_pred.append(joblib.load(exppath / f"fold-{i}_preds"))
    return y_test, y_pred

def lazy_load_folds(exppath, n_folds=10):
    for i in range(n_folds):
        y_test = joblib.load(exppath / f"fold-{i}_true")
        y_pred = joblib.load(exppath / f"fold-{i}_preds")
        meta = joblib.load(exppath / f"fold-{i}_metadata")
        try:
            y_aug = joblib.load(exppath / f"fold-{i}_preds_augmented")
            yield y_test, y_pred, y_aug, meta
        except FileNotFoundError:
            yield y_test, y_pred, meta

def get_labels(dataset):
    tube_labels = dict()
    for tube in "A", "B", "C":
        arbitrary_patient = next(iter(dataset.fc_dataset.values()))
        pns_labels = copy.copy(arbitrary_patient[tube].pns_labels)
        for bad_markers in "TIME", "":
            try:
                pns_labels.remove(bad_markers)
            except ValueError:
                pass
        tube_labels[tube] = pns_labels
    return tube_labels

def tensor_to_df(features, dataset, rescale=True):
    frames = []
    if rescale:
        cells = dataset.scaler.inverse_transform(features).squeeze()
    else:
        cells = features.squeeze()

    for tube_idx, (tube, markers) in enumerate(get_labels(dataset).items()):
        df = pd.DataFrame(cells[tube_idx], columns=markers)
        df["tube"] = tube
        frames.append(df)
    return pd.concat(frames)

def df_to_tensor(df):
    input = []
    for tube, tube_df in df.groupby("tube"):
        input.append(tube_df.dropna(axis=1).drop("tube", axis=1).values)
    input = torch.tensor(np.stack(input, axis=0))
    return input

