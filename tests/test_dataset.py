"""
Unit tests for the CytometryDataset class and related utilities.

This module tests:
- Coherence between dataset labels and metadata.
- Inclusion and shape of tabular metadata features.
- Correct scaling and standardization of cytometry and tabular features
  after train/validation split
"""

from pathlib import Path
import math
import pytest
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # For debugging


def test_metadata_label_coherence(dataset):

    if dataset.return_patient_id:
        dl = DataLoader(dataset, batch_size=1)
        meta = dataset.metadata
        target_col = dataset.target_col
        mapping = dataset.label_mapping
        for batch in dl:
            row = batch[-1]
            labels = batch[1]
            patient_row = meta.loc[meta.Patient == row[0], :].squeeze()
            status = patient_row[target_col]
            status_label = mapping[status]
            assert status_label == int(labels)


def test_tabular_features(ds_tabular):
    dl = DataLoader(ds_tabular, batch_size=1)
    output = next(iter(dl))
    assert len(output) == 4


def test_dataset_scaling(dataset_100_cells):
    """
    Perform a train-val split with
    StratifiedShuffleSplit and CytoSubset, fit
    scaler on train and assert that the validation
    subset is correctly standardized.
    """
    from flowcyt.dataset import CytoSubset
    from flowcyt.scaling import CytoScaler
    from sklearn.model_selection import StratifiedKFold, KFold
    import numpy as np
    from scipy.stats import wilcoxon

    dataset = dataset_100_cells
    if dataset.target_col == "FLT3_ratio":
        Splitter = KFold # Cannot stratify easily on regression tasks
    else:
        Splitter = StratifiedKFold

    # Prepare stratification labels
    y = dataset.metadata[dataset.target_col].map(dataset.label_mapping).to_numpy()
    X = np.zeros(len(dataset))  # Dummy X for split

    splitter = Splitter(n_splits=2)
    train_idx, val_idx = next(splitter.split(X, y))

    train_set = CytoSubset(dataset, train_idx)
    val_set = CytoSubset(dataset, val_idx)

    # Fit scaler on train, set on val
    train_set.fit_scaler(CytoScaler)
    assert val_set.scaler is None
    val_set.set_scaler(train_set.scaler)

    # Check scaling on validation set
    dl = DataLoader(val_set, batch_size=len(val_set))
    batch = next(iter(dl))
    cyto_features = batch[0]

    cyto_mean = cyto_features.mean(axis=(0, 2)).flatten()
    means = [cyto_mean]
    cyto_std = cyto_features.std(axis=(0, 2)).flatten()
    stds = [cyto_std]

    if dataset.tabular_features is not None:
        tab_features = batch[2]
        tab_mean = tab_features.mean(axis=0).flatten()
        tab_std = tab_features.std(axis=0).flatten()
        means.append(tab_mean)
        stds.append(tab_std)

    pvalues = []
    for mean in means:
        pvalues.append(
            wilcoxon(
                mean
            ).pvalue
        )
    for std in stds:
        pvalues.append(
            wilcoxon(
                std - 1
            ).pvalue
        )
    assert all(
        map(lambda pval: pval>=0.001, pvalues)
    ), "Data is not properly scaled"
