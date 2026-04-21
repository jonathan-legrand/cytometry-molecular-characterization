import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import copy
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from flowcyt.cnn import CytoNet, AttentionCytoNet, ModelWrapper
from flowcyt.dataset import LABEL_MAPPING, CytoSubset
from flowcyt.inference.inference import get_fold_path, load_best, plot_resample_preds, tune_threshold
from flowcyt.utils import name_to_dict

def get_run_config(expname):
    dct = name_to_dict(expname)
    config_name = dct["configname"]
    with open(f"run_config/{config_name}.json", "r") as stream:
        run_config = json.load(stream)
    return run_config

def gen_model_and_test(
        expname,
        dataset,
        return_thresh=False,
        ModelClass=ModelWrapper,
        tabular_features=None
    ):
    
    models_paths = Path("models") / expname
    fold_paths = tuple(models_paths.iterdir())
    
    run_config = get_run_config(expname)
    model_spec = run_config["model_spec"]

    sfk = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=2025
        )

    mock_X = np.zeros(len(dataset))
    y = dataset.metadata[dataset.target_col].map(dataset.label_mapping).to_numpy()

    assert dataset.scaler is None

    for i, (temp_idx, test_idx) in enumerate(sfk.split(mock_X, y)):
        set_copy = copy.copy(dataset)
        test_set = CytoSubset(set_copy, test_idx)
        temp_set = CytoSubset(dataset, temp_idx)
    
        model_dir = get_fold_path(fold_paths, i)
        state_dict, scaler_path = load_best(model_dir)
        scaler = joblib.load(scaler_path)

        test_set.set_scaler(scaler)
        temp_set.set_scaler(scaler)

        model = ModelClass(**model_spec, tabular_features=tabular_features).model
        model.load_state_dict(state_dict)
        model.eval()

        if return_thresh:
            thresh_tuple = tune_threshold(model, temp_set)
            yield i, model, test_set, *thresh_tuple
        else:
            yield i, model, test_set

