"""
Dark magic so that I can import packages from flowcyt
"""
from pathlib import Path
import sys
import os
import json

import pytest
import torch
from torch import nn
from torch import optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flowcyt.cnn import AttentionNetWithMetadata, ModelWrapper
from flowcyt.utils import get_config
from flowcyt.dataset import CytometryDataset, tabular_features


@pytest.fixture(scope="module")
def config():
    config = get_config()
    return config

@pytest.fixture(
        scope="module",
        params=["depthwiseconvtesting", "tabulartesting"]
    )
def run_config(request):
    config_name = request.param
    with open(f"run_config/{config_name}.json", "r") as stream:
        run_config = json.load(stream)
    return run_config

@pytest.fixture(scope="module")
def ds_tabular(config):
    datapath = Path(config["FCS_PATH"])
    dataset = CytometryDataset(
        datapath,
        n_cells=10,
        return_patient_id=True,
        resample_cells=False,
        tabular_features=tabular_features,
        target_col="NPM",
    )
    return dataset

dataset_config = [
    {"return_patient_id": False, "target_col": "NPM"},
    {"return_patient_id": True, "target_col": "FLT3"},
]

@pytest.fixture(scope="module", params=dataset_config)
def dataset(request, config, run_config):
    datapath = Path(config["FCS_PATH"])
    
    dataset = CytometryDataset(
        datapath,
        n_cells=10,
        resample_cells=False,
        tabular_features=run_config.get("tabular_features", None),
        **request.param,
    )
    return dataset

@pytest.fixture(scope="module", params=dataset_config)
def dataset_100_cells(request, config, run_config):
    datapath = Path(config["FCS_PATH"])
    
    dataset = CytometryDataset(
        datapath,
        n_cells=100,
        resample_cells=False,
        tabular_features=run_config.get("tabular_features", None),
        **request.param,
    )
    return dataset

@pytest.fixture(scope="module")
def model_wrapper(run_config):
    """
    Instantiate the model and dummy train it to
    make results more realistic
    """
    model_spec = run_config["model_spec"]
    batch_size = run_config["batch_size"]
    features_shape = (
        batch_size,
        model_spec["n_tubes"],
        model_spec["n_cells"],
        model_spec["n_markers"],
    )
    tabular_features = run_config.get("tabular_features", None)

    model_wrapper = ModelWrapper(tabular_features, **model_spec)
    model = model_wrapper.model
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-1) # Huge lr to really move parameters
    model.train()
    for _ in range(20):
        features = torch.rand(features_shape)
        labels = torch.randint(0, 2, (batch_size, 1)).float()

        if isinstance(model, AttentionNetWithMetadata):
            tabular = torch.rand((batch_size, model.n_tabular_variables))
        else:
            tabular = None

        logits = model_wrapper(features, tabular)
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    model.eval()

    return model_wrapper