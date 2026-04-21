"""
Training Script for Cytometry Neural Network

This script trains a neural network model for cytometry data classification using
cross-validation. It supports multiple folds, early stopping, and various
augmentation techniques.

The trained model is a multitube adaptation of the model presented in the following paper:
Hu, Z., Tang, A., Singh, J., Bhattacharya, S., & Butte, A. J. (2020). A robust and
interpretable end-to-end deep learning model for cytometry data. Proceedings of the
National Academy of Sciences of the United States of America, 117(35), 21373.
https://doi.org/10.1073/pnas.2003026117

The script performs the following main steps:
1. Load configuration and dataset
2. Perform 10-fold stratified cross-validation
3. For each fold: train model, validate, and evaluate on test set
4. Save model checkpoints, predictions, and evaluation metrics
5. Log training progress using TensorBoard

Usage:
    python train_network.py <target_column> <config_name> <prefix>

Arguments:
    target_column: The column name in the dataset to predict
    config_name: Name of the JSON config file in run_config/ directory
    prefix: Prefix for logging and output directories

Output:
    - Trained models in models/ directory
    - Predictions and metadata in prediction/ directory
    - TensorBoard logs in runs/ directory
"""

from datetime import datetime
import os
import sys
import json
import copy
from flowcyt.inference.inference import test_time_augmented_inference
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import average_precision_score

from flowcyt.cnn import count_parameters, EarlyStopper
from flowcyt.dataset import CytometryDataset, CytoSubset
from flowcyt.cnn import ModelWrapper
from flowcyt.scaling import CytoScaler
from flowcyt.utils import get_config, build_name
from flowcyt.inference import dataset_inference
from flowcyt.training import print_training_params, train_epoch, evaluate_epoch, parse_batch

config = get_config()
torch.manual_seed(2025)

# Run vars
DATAPATH = Path(config["FCS_PATH"])
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"


torch.use_deterministic_algorithms(False)


def train_net(model_wrapper, training_set, val_set, fold_idx, logname, **run_config):

    print(run_config)
    model = model_wrapper.model
    
    # Data loading
    train_loader = DataLoader(
        training_set,
        batch_size=run_config["batch_size"],
        shuffle=True,
        num_workers=8,
        prefetch_factor=2
    )
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    # Define model, loss and optim
    _, counts = np.unique(y, return_counts=True)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(counts[0] / counts[1])
    )

    model = model.to(device)
    print(f"Training on {device}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        **run_config["optimizer_params"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        **run_config["scheduler_params"]
    )

    # Init logging, checkpointing, early stopping
    fname = f"fold-{fold_idx}"
    os.makedirs(Path("runs") / run_config["expname"] / logname, exist_ok=True)
    writer_path = Path("runs") / run_config["expname"] / logname / fname
    writer = SummaryWriter(writer_path)
    print(f"Writing loss curves to {writer_path}")
    model_dir = Path("models") / logname / fname
    os.makedirs(model_dir, exist_ok=False)

    best_vloss = np.inf
    stopper = EarlyStopper(
        patience=run_config["patience"],
        min_delta=0.05
    )

    # Training loop
    for epoch in range(run_config["epochs"]):
        print(f"FOLD {fold_idx} EPOCH {epoch + 1}")

        model.train(True)
        avg_loss = train_epoch(
            model_wrapper,
            train_loader,
            optimizer,
            loss_fn,
            device=device
        )

        # Compute and log val metrics
        avg_vloss, avg_vacc = evaluate_epoch(
            model_wrapper, val_loader, loss_fn, device
        )
        print(
            f'Epoch {epoch + 1} : LOSS train {avg_loss} valid {avg_vloss} valid score {avg_vacc} lr {scheduler.get_last_lr()}',
            end=" "
        )
        scheduler.step(avg_vloss)

        # Logging
        writer.add_scalars(
            f'fold-{fold_idx}_learning-curves',
            {
                'train' : avg_loss,
                'eval' : avg_vloss,
                'classification_score' : avg_vacc,
                }, epoch + 1)

        if epoch % 10 == 0: # This is time consuming, so do not call it every epoch
            for i, (name, param) in enumerate(model.named_parameters()):
                writer.add_histogram(f"{i}_{name}", param, global_step=epoch + 1)
        writer.flush()

        # Checkpointing
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = model_dir / f"fold-{fold_idx}_epoch-{epoch}"
            torch.save(model.state_dict(), model_path)

        # Early stopping
        print("Counter value :", stopper.counter)
        if stopper.check(avg_vloss) is True:
            stopper.log()
            break


    writer.close()
    return model_path


if __name__ == "__main__":

    target_col = sys.argv[1]
    config_name = sys.argv[2]
    prefix = sys.argv[3]

    with open(f"run_config/{config_name}.json", "r") as stream:
        run_config = json.load(stream)

    model_spec = run_config["model_spec"]
    tabular_features = run_config.get("tabular_features", None)

    dataset = CytometryDataset(
        DATAPATH,
        n_cells=run_config["model_spec"]["n_cells"],
        target_col=target_col,
        tabular_features=tabular_features
    )
    
    # Actual model will be instantiated later, this is just for
    # logging and weight ini
    dummy_model = ModelWrapper(tabular_features, **model_spec).model
    nparams = count_parameters(dummy_model)
    print("|params| = ", nparams)
    print(dummy_model)
    logname = build_name(
        prefix,
        predict=target_col,
        configname=config_name,
        nparams=nparams,
        stamp=timestamp
    )
    print(logname)

    logpath = Path("prediction") / logname
    os.makedirs(logpath, exist_ok=False)
    torch.save(
        dummy_model.state_dict(),
        "/tmp/initial_weights.pth"
    )

    # Required for stratification
    mock_X = np.zeros(len(dataset))
    y = dataset.metadata[dataset.target_col].map(dataset.label_mapping).to_numpy()
    print_training_params(mock_X, y)
    
    sfk = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=2025
    )
    test_scores = []

    for i, (temp_idx, test_idx) in enumerate(sfk.split(mock_X, y)):
        model_wrapper = ModelWrapper(tabular_features, **run_config["model_spec"])
        model = model_wrapper.model
        model.load_state_dict(
            torch.load("/tmp/initial_weights.pth", weights_only=True)
        )

        set_copy = copy.copy(dataset)
        set_copy.return_patient_id = True # Needed for bad classification report
        test_set = CytoSubset(set_copy, test_idx)
        dataset.resample_cells = run_config["resample_training"]
        temp_set = CytoSubset(dataset, temp_idx)

        # Further split for validation
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=2025,
        )
        
        train_idx, val_idx = next(
            splitter.split(mock_X[temp_idx], y[temp_idx])
        )
        val_set = CytoSubset(copy.copy(temp_set), val_idx)
        train_set = CytoSubset(temp_set, train_idx)

        train_set.fit_scaler(CytoScaler)
        val_set.set_scaler(train_set.scaler)
        test_set.set_scaler(train_set.scaler)
        
        # Augmentation by resampling
        best_model_path = train_net(
            model_wrapper,
            train_set,
            val_set,
            fold_idx=i,
            logname=logname,
            **run_config
        )

        state_dict = torch.load(best_model_path, weights_only=True)
        model_wrapper = ModelWrapper(tabular_features, **model_spec)
        model = model_wrapper.model
        model.load_state_dict(state_dict)

        inference_result = dataset_inference(model_wrapper, test_set)
        if len(inference_result) == 3:
            y_test, y_pred, rows = inference_result
        else:
            y_test, y_pred = inference_result
            rows = None
        test_set.resample_cells = True

        print_training_params(mock_X[test_idx], y_test)
        joblib.dump(y_test, logpath / f"fold-{i}_true")
        joblib.dump(y_pred, logpath / f"fold-{i}_preds")
        joblib.dump(rows, logpath / f"fold-{i}_metadata")

        test_score = average_precision_score(y_test, y_pred)
        if tabular_features is None:
            y_test_, y_pred_augmented = test_time_augmented_inference(model, test_set)
            joblib.dump(y_pred_augmented, logpath / f"fold-{i}_preds_augmented")
            test_score_aug = average_precision_score(y_test, y_pred_augmented)
            print(f"Test score = {test_score:.2f}, augmented = {test_score_aug:.2f}")
        else:
            print(f"Test score = {test_score:.2f}")

        train_set.store_scaler(best_model_path.parent / "scaler.joblib")

        test_scores.append(test_score)
        
    print(test_scores)
    print(f"Predictions logged in {logpath}")

# %%
