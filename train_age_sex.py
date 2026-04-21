from pathlib import Path
from datetime import datetime
import os
import sys

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from flowcyt.dataset import CytometryDataset, fetch_X_y_meta
from flowcyt.utils import get_config, build_name
from flowcyt.evaluation import FoldResults

rng = np.random.default_rng(seed=1234)

# Init parameters and logging
config = get_config()
DATAPATH = Path(config["FCS_PATH"])
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define pipeline
transformers = [
    ("sexencoder", OneHotEncoder(), ("sexe",)),
    ("pass", ("passthrough"), ("age", "Blastes moelle osseuse (%)")),
]
ct = ColumnTransformer(
   transformers, remainder="drop"
)

pipe = make_pipeline(
    ct,
    RandomForestClassifier(
        class_weight="balanced",
        random_state=1234
    )
)

sfk = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=2025
)

if __name__ == "__main__":
    target_col = sys.argv[1]

    # Prepare logging
    logname = build_name(
    "age, sex, blast percentage",
    predict=target_col,
    stamp=timestamp
    )
    logpath = Path("prediction") / logname
    os.makedirs(logpath, exist_ok=True)

    dataset = CytometryDataset(
        DATAPATH,
        n_cells=10,
        return_patient_id=True,
        resample_cells=False,
        target_col=target_col,
        scaler=None
    )
    X, y, metadata = fetch_X_y_meta(dataset)


    # Training and evaluation
    cv_results = []
    for i, (train_idx, test_idx) in enumerate(sfk.split(X, y)):

        pipe.fit(metadata.iloc[train_idx, :], y[train_idx])
        y_score = pipe.predict_proba(metadata.iloc[test_idx, :])[:, 1]

        y_test = y[test_idx]
        rows = metadata.iloc[test_idx, :].Patient.values
        print(rows)

        # Store predictions for plot_results.ipynb
        joblib.dump(y_test, logpath / f"fold-{i}_true")
        joblib.dump(y_score, logpath / f"fold-{i}_preds")
        joblib.dump(rows, logpath / f"fold-{i}_metadata")


        # Compute results for visualisation below
        fr = FoldResults(y[test_idx], y_score, fold=i)
        perfs = fr.compute_perfs()
        perfs.append(
            {
                "fold": i,
                "metric": roc_auc_score.__name__,
                "score": roc_auc_score(y_test, y_score)
            }
        )
        cv_results.append(pd.DataFrame(perfs))

    cv_results = pd.concat(cv_results)
    print(f"Predictions exported to {logpath}")


    # Plot results
    plt.subplots(figsize=(10, 5))
    sns.boxplot(cv_results, y="score", hue="metric", fill=False)
    sns.stripplot(cv_results, y="score", hue="metric", dodge=True, jitter=False, legend=False)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.show()


