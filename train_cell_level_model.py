# %%
from pathlib import Path
from datetime import datetime
import os
import sys

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, precision_recall_curve, roc_auc_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.ensemble import RandomForestClassifier

from flowcyt.dataset import CytometryDataset, fetch_X_y_meta
from flowcyt.utils import get_config, build_name
from flowcyt.evaluation import METRICS, FoldResults, precision_at_k, precision_at_min_recall
from flowcyt.mil_classifier import PatientPredictor, PatientPredictorWithClinical
from flowcyt.scaling import CytoScaler
from flowcyt.plotting import attribution_palette, tree_to_gating_max_path

rng = np.random.default_rng(seed=1234)

# Init parameters and logging
config = get_config()
DATAPATH = Path(config["FCS_PATH"])
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
WITH_TABULAR = False

cols = ("age", "Blastes moelle osseuse (%)")

sfk = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=2025
)

if __name__ == "__main__":
    target_col = sys.argv[1]
    n_cells = eval(sys.argv[2])
    try:
        make_plots = eval(sys.argv[3]) 
        assert isinstance(make_plots, bool)
    except IndexError:
        make_plots = False

    dataset = CytometryDataset(
        DATAPATH,
        n_cells=n_cells,
        return_patient_id=True,
        resample_cells=False,
        target_col=target_col,
        scaler=None
    )
    X, y, metadata = fetch_X_y_meta(dataset)
    
    if WITH_TABULAR:
        patient_clf = PatientPredictorWithClinical(
            cell_predictor=make_pipeline(StandardScaler(), DecisionTreeClassifier(min_samples_leaf=1000)),
        )
    else:
        patient_clf = make_pipeline(
            CytoScaler(),
            PatientPredictor(
                cell_predictor=DecisionTreeClassifier(min_samples_leaf=1000),
                tube_pooling_func=np.max
            )
        )
        
    # Prepare logging
    cell_clf_name = patient_clf.named_steps["patientpredictor"].cell_predictor.__class__.__name__
    
    logname = build_name(
        "cell-level-model",
        predict=target_col,
        ncells=n_cells,
        withtab=WITH_TABULAR,
        clf=cell_clf_name,
        stamp=timestamp
    )
    print(logname)

    logpath = Path("prediction") / logname
    os.makedirs(logpath, exist_ok=True)

    # Training and evaluation
    # Scaling happens outside the pipeline
    # because other dataset will have different
    # unscaled value. We want our pipeline to just expect
    # zscored values
    for i, (train_idx, test_idx) in enumerate(sfk.split(X, y)): # Scaling should happen around here

        X_train = X[train_idx, ...]

        X_test = X[test_idx, ...]
        y_test = y[test_idx]
        
        if WITH_TABULAR:
            train_features = metadata.iloc[train_idx].loc[:, cols] # I hate that
            train_features["cytometry"] = list(X_train)

            test_features = metadata.iloc[test_idx].loc[:, cols]
            test_features["cytometry"] = list(X_test)

            patient_clf.fit(train_features, y[train_idx])
            y_score = patient_clf.predict_proba(test_features)[:, 1]
        else:
            patient_clf.fit(X_train, y[train_idx])
            y_score = patient_clf.predict_proba(X_test)[:, 1]

        rows = metadata.iloc[test_idx, :].Patient.values
        print(rows)
        
        # Store predictions for plot_results.ipynb
        joblib.dump(y_test, logpath / f"fold-{i}_true")
        joblib.dump(y_score, logpath / f"fold-{i}_preds")
        joblib.dump(rows, logpath / f"fold-{i}_metadata")

    print(f"Predictions exported to {logpath}")

    # This is for evaluation on external dataset
    # so we perform hyperparameter search and tuning here
    
    ## Grid search
    param_grid = {
        # Decision tree parameters
        "patientpredictor__cell_predictor__max_depth": [3, 4],
        "patientpredictor__cell_predictor__min_samples_leaf": [1, 100, 500],
        "patientpredictor__cell_predictor__ccp_alpha": np.logspace(0, -6, 4),

        # Tube pooling function
        "patientpredictor__tube_pooling_func": [np.max, np.mean, np.min],
    }

    # Grid scorer;
    grid = GridSearchCV(
        estimator=patient_clf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    
    grid.fit(X, y)
    best_patient_clf = grid.best_estimator_
    pd.DataFrame(grid.cv_results_).to_csv(f"output/grid_search_{logname}.csv")

    ## For thresh tuning we have two options
    ## fbeta score with beta < 1 => Precision more important than recall 
    ## or specify the min recall and optimize precision with that constraint
    ## Min recall should not be to low though because those thresholds 
    ## can get stupidly small
    #scorer = make_scorer(
    #    precision_at_min_recall,
    #    min_recall=0.2
    #)
    scorer = make_scorer(
        fbeta_score,
        beta=0.5
    )

    tuned_clf = TunedThresholdClassifierCV(
        estimator=best_patient_clf,
        refit=True,
        n_jobs=-1,
        random_state=1234,
        thresholds=100,
        scoring=scorer
    )
    if WITH_TABULAR:
        features = metadata.loc[:, cols]
        features["cytometry"] = list(X)
        tuned_clf.fit(features, y)
        lr = tuned_clf.estimator.named_steps["patientpredictorwithclinical"].aggregator_.named_steps["logisticregression"]
        print(dict(zip(["Age", "Blast %", "A", "B", "C"], list(lr.coef_[0]))))
    else:
        tuned_clf.fit(X, y)
    # Store for evaluation on external dataset
    clf_export_path = Path("sklearn-models") / (logname + ".joblib")
    joblib.dump(tuned_clf, clf_export_path)
    print("Classifier exported to", clf_export_path)
    print(f"Best thresh :", tuned_clf.best_threshold_)
    
    if make_plots:
        # Let's show the tree on the training set
        tubes = ("A", "B", "C")
        depth = 3
        fig, axes = plt.subplots(depth, len(tubes), figsize=(15, 5 * depth))

        for idx, tube in enumerate(tubes):
            dfs = [dataset.get_df(i)[0][idx] for i in range(len(dataset))]
            dfs = pd.concat(dfs, axis=0) #scaling

            feature_names = dfs.columns
            dfs = (dfs - scaler.mean_.squeeze()[idx]) / scaler.scale_.squeeze()[idx]
            print(feature_names)

            tube_predictor = tuned_clf.estimator.tube_predictors_[idx]
            tree = tube_predictor.cell_predictor.named_steps["decisiontreeclassifier"]

            hue = tree.predict_proba(dfs)[:, 1]
            tree_to_gating_max_path(tree, dfs, axes[:, idx], hue=hue, feature_names_in=feature_names)
        plt.show()
    

# %%
