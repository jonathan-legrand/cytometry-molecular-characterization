from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from natsort import natsorted
from pathlib import Path

from flowcyt.training import parse_batch

confusion_mapping = (
    {"mutated": "FP", "WT": "TN"},
    {"mutated": "TP", "WT": "FN"}
)

def get_fold_path(it, fold_idx):
    for path in it:
        if int(path.name.split("-")[1]) == fold_idx:
            return path
    return None

def load_best(model_dir: Path):
    assert isinstance(model_dir, Path)
    paths = natsorted(list(model_dir.iterdir()))
    bestpath = paths[-2]
    scaler_path = paths[-1] # I hate this
    print(bestpath)
    state_dict = torch.load(
        bestpath,
        weights_only=True,
    )
    return state_dict, scaler_path

def dataset_inference(model, test_set):
    test_loader = DataLoader(test_set, batch_size=len(test_set), pin_memory=True)
    model.eval()
    y_test = []
    y_pred = []
    rows = []
    for test_data in test_loader:
        if test_set.return_patient_id:
            inputs, labels, row = parse_batch(test_data, test_loader, return_id=True)
            rows.append(row)
        else:
            inputs, labels = test_data
            
        with torch.no_grad():
            logits = model(*inputs)
        probs = F.sigmoid(logits).cpu().numpy()
        
        y_pred.append(probs.squeeze())
        y_test.append(labels.numpy().squeeze())
    y_test, y_pred = np.concatenate(y_test), np.concatenate(y_pred)
    if test_set.return_patient_id:
        return y_test, y_pred, row
    else:
        return y_test, y_pred

def soft_inference(model, inputs):
    with torch.no_grad():
        logits = model(torch.tensor(inputs, dtype=torch.float32))
        probs = F.sigmoid(logits).numpy()
    return probs.squeeze()

def test_time_augmented_inference(
        model,
        test_set,
        n_res=30,
        agg_func=np.mean
    ):
    assert test_set.resample_cells
    model.eval()
    y_test, y_score = [], []
    for i in range(len(test_set)):
        preds_same_sample = []
        for _ in range(n_res):
            if test_set.return_patient_id:
                f, l, _ = test_set[i]
            else:
                f, l = test_set[i]

            with torch.no_grad():
                logit = model(f.reshape(1, *f.shape)).squeeze()

            prob = F.sigmoid(logit).cpu().numpy()
            preds_same_sample.append(prob)
        l = int(l.cpu())
        
        y_test.append(l)
        y_score.append(agg_func(preds_same_sample))
    return y_test, y_score

# TODO Add if correct classification (color title?)

colors = ["blue", "red"]
from sklearn.metrics import roc_auc_score
def plot_resample_preds(model, dset, n_res=50):
    assert dset.resample_cells
    model.eval()
    dl = DataLoader(dset)
    fig, ax = plt.subplots()
    y_test, y_score = [], []
    for i in range(len(dset)):
        preds_same_sample = []
        for _ in range(n_res):
            batch = dset[i]
            batch = parse_batch(batch, dl, return_id=False)
            f, l = batch
            
            # Shape black magic
            f = list(f)
            f[0] = f[0].reshape((1, *f[0].shape))
            if len(f) > 1:
                f[1] = f[1].reshape((1, *f[1].shape))

            with torch.no_grad():
                logit = model(*f).squeeze()
            prob = F.sigmoid(logit).cpu().numpy()
            preds_same_sample.append(prob)
        l = int(l.cpu())
        ax.hist(
            preds_same_sample,
            alpha=0.1,
            color=colors[l]
        )
        y_test.append(l)
        y_score.append(np.mean(preds_same_sample))
    auc = roc_auc_score(y_test, y_score)
    ax.set_xlim(0, 1)
    ax.set_title(f"Resampled predictions, {i} subjects, ROCAUC = {auc:.2f}")
    return fig, ax, y_test, y_score


from sklearn.metrics import (
    balanced_accuracy_score,
    average_precision_score,
    precision_score
)

def tune_threshold(model, tuning_set):
    y_true, y_pred, _ = dataset_inference(model, tuning_set)
    threshs = [i/20 for i in range(20)]
    ba = []
    for thresh in threshs:
        y_decision = np.where(y_pred > thresh, 1, 0)
        ba.append(balanced_accuracy_score(y_true, y_decision))
    idx_opt = np.argmax(ba)
    thresh_opt = threshs[idx_opt]

    fig, ax = plt.subplots()
    ax.plot(threshs, ba)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Balanced accuracy")
    ax.scatter(thresh_opt, ba[idx_opt], color="red")
    ax.set_title(f"Optimal threshold on validation set : {thresh_opt}")
    return thresh_opt, fig, ax

