# %%
import sys
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy.stats import zscore
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.tree import plot_tree
from sklearn.tree._export import _MPLTreeExporter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from statannotations.Annotator import Annotator

from flowcyt.dataset import TestSet, pns_dict_camilla, fetch_X_y_meta
from flowcyt.utils import get_config, name_to_dict
from flowcyt.plotting import attribution_palette, tree_to_gating_max_path
from flowcyt.scaling import CytoScaler
from flowcyt.evaluation import npv_score

config = get_config()
DATAPATH = Path(config["TEST_FCS"])
EXCEL_PATH = Path(config["TEST_META"])
PLOT_TREES = True
DEBUG = False

cmap = attribution_palette
low = cmap(0.0)
high = cmap(1.0)
colors = [low, high]
tubes = ("A", "B", "C")
plots_path = Path(config["PLOT_PATH"])
disp_tubes = tubes # Controls which trees we want to plot

def replace_text(obj):
        import re
        if type(obj) == mpl.text.Annotation:
            txt = obj.get_text()
            txt = re.sub("samples","$K$",txt)
            txt = re.sub("\nvalue.*","",txt)
            txt = re.sub("<=","\n<=",txt)
            obj.set_text(txt)
        return obj

    
def custom_fill_color(
        tree,
        node_id,
        saturation_factor=0.6,
        threshold=0.5
    ):
    value = tree.value[node_id][0]
    value /= value.sum()

    if value[1] > threshold:
        hsv_high = mpl.colors.rgb_to_hsv(high[:-1])
        hsv_high[1] = value[1] * saturation_factor
        mapped_color = mpl.colors.hsv_to_rgb(hsv_high) 
    else:
        hsv_low = mpl.colors.rgb_to_hsv(low[:-1])
        hsv_low[1] = value[0] * saturation_factor
        mapped_color = mpl.colors.hsv_to_rgb(hsv_low) 
    return mapped_color


if __name__ == "__main__":
    if DEBUG:
        print("WARNING DEBUG MODE")
        ##estimator_name = "cell-level-model_predict-NPM_ncells-100_withtab-False_clf-decisiontreeclassifier_pooling-max_stamp-20260114_114727"
        estimator_name = "cell-level-model_predict-FLT3_ncells-100_withtab-False_clf-DecisionTreeClassifier_stamp-20260408_143648"
        n_seeds = 2
    else:
        estimator_name = sys.argv[1]
        try:
            n_seeds = eval(sys.argv[2])
        except IndexError:
            n_seeds = 1

    # Load models
    estimator_path = Path(
        "sklearn-models"
    ) / (estimator_name + ".joblib")
    tuner = joblib.load(estimator_path)
    estimator = tuner.estimator_
    estimator_dct = name_to_dict(estimator_name)
    target_col = estimator_dct["predict"]
    n_cells = eval(estimator_dct["ncells"])

    # Compute prediction scores
    scores = []
    # Compute scores multiple times with different subsampling seeds
    # to obtain mean [SD] estimates
    for seed in range(n_seeds):
        dataset = TestSet(
            DATAPATH,
            mpath=EXCEL_PATH,
            n_cells=n_cells,
            return_patient_id=True,
            resample_cells=True,
            target_col=target_col,
            random_state=seed,
            tubes=tubes
        )

        X, y, meta = fetch_X_y_meta(dataset, raise_shape=False)
    
        y_pred = tuner.predict(X)
        y_proba = tuner.predict_proba(X)

        # Patient level scores
        precision = precision_score(y, y_pred)
        npv = npv_score(y, y_pred)
        roc = roc_auc_score(y, y_proba[:, 1])
        realised_scores = dict(AUROC=roc, PPV=precision, NPV=npv, tube="all")
        scores.append(realised_scores)
        
        # Tube level scores
        try:
            X_scaled = estimator.named_steps["cytoscaler"].transform(X)
        except KeyError:
            print("No scaler found, keep raw values")
            X_scaled = X
        stacked_probas = estimator.named_steps["patientpredictor"].compute_stacked_probas(X_scaled)
        for idx, tube in enumerate(tubes):
            print(f"Tube {tube}")
            tube_probas = stacked_probas[idx]
            tube_preds = np.where(tube_probas > tuner.best_threshold_, 1, 0)
            tube_scores = dict(tube=tube)
            tube_scores["AUROC"] = roc_auc_score(y, tube_probas)
            tube_scores["PPV"] = precision_score(y, tube_preds)
            tube_scores["NPV"] = npv_score(y, tube_preds)
            scores.append(tube_scores)

        scores.append(realised_scores)
    res = pd.DataFrame(scores)
    agg = res.groupby("tube").agg(["mean", "std"]).round(decimals=2)
    
    # Format as rounded strings in mean [SD] format for export
    result = pd.DataFrame()
    for col in ["AUROC", "PPV", "NPV"]:
        result[col] = (
            agg[(col, "mean")].round(3).astype(str)
            + " ["
            + agg[(col, "std")].round(3).astype(str)
            + "]"
        )
    result.to_csv(f"output/inference-scores_predict-{target_col}_ncells-{n_cells}_nseeds-{n_seeds}.csv")


    # Show histogram of predictions and tuned threhsold
    h_1 = plt.hist(y_proba[y == 0, 1], alpha=0.5)
    h_2 = plt.hist(y_proba[y == 1, 1], alpha=0.5)
    y_max = max(h_1[0].max(), h_2[0].max())
    plt.vlines(tuner.best_threshold_, 0, y_max, colors="tab:red")
    plt.title(f"Precision = {precision:.2f}")
    plt.show()

    # Display per tube pvalues
    stacked_probas = pd.DataFrame(stacked_probas.T, columns=[f"Tube {letter}" for letter in ("A", "B", "C")])
    stacked_probas["Mutation status"] = np.where(y==1, "Mutated", "Wild Type")
    stacked_probas.melt(id_vars="Mutation status")
    m = stacked_probas.melt(
        id_vars="Mutation status",
        var_name="Decision method",
        value_name="Patient score",
    )
    fig, ax = plt.subplots(figsize=(9, 6))

    fig_args = dict(
        x="Decision method",
        hue="Mutation status",
        y="Patient score",
        palette=colors,
        hue_order=["Wild Type", "Mutated"],
        ax=ax,
        dodge=True,
    )

    sns.stripplot(
        m,
        legend=None,
        **fig_args,
    )
    sns.boxplot(
        m,
        fill=None,
        legend=None,
        flierprops={"marker": None},
        **fig_args
    )
    plt.grid()
    plt.ylabel("Mean of cells score ($\\hat{z}$)", size=18, rotation=90)
    plt.xlabel(None)
    if target_col == 'FLT3':
        letter = "B"
    else:
        letter = "A"
    ax.text(
        -0.06,
        1.1,
        letter,
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='left'
    )
    ax.tick_params(labelsize=15)

    comparisons = [((method, "Mutated"), (method, "Wild Type")) for method in m["Decision method"].unique()]

    configuration = {
        'test':'Mann-Whitney-ls',
        'text_format':'star',
    }

    annotator = Annotator(
        pairs=comparisons,
        data=m,
        **fig_args
    )
    annotator.configure(**configuration).apply_test(
    ).annotate()
    plt.savefig(plots_path / f"fig-tubes_predict-{target_col}_ncells-{n_cells}.png")
    
    # Export classification df
    meta["z_hat"] = y_proba[:, 1]
    meta["y_pred"] = y_pred
    meta["y_test"] = y
    output_path = Path("output") / (estimator_name + "classification.csv")
    meta.to_csv(output_path)
    print("Classification df exported to ", output_path)
    
    # Plot decision trees
    if PLOT_TREES and estimator_dct["clf"].lower() == "decisiontreeclassifier":
        
        depth = 2
        fig, axes = plt.subplots(depth, len(tubes), figsize=(15, 5 * depth))
        add_cbar=True
        norm = mpl.colors.CenteredNorm(vcenter=tuner.best_threshold_)

        for idx, tube in enumerate(tubes):
            axes[0, idx].set_title(f"Tube {tube}", size=20)
            dfs = []
            for i in range(len(dataset)):
                batch = dataset.get_df(i)
                df = batch[0][idx]
                dfs.append(df)

            dfs = pd.concat(dfs, axis=0)
            disp_msk = dfs.iloc[:, 2] <= 16 # Remove channel 2 outliers
            dfs = dfs[disp_msk]
            dfs = (dfs - dfs.mean(axis=0)) / dfs.std(axis=0)

            feature_names = dfs.columns
            print(feature_names)

            patient_predictor =  estimator.named_steps["patientpredictor"]
            tube_predictor = patient_predictor.tube_predictors_[idx]
            tree = tube_predictor.cell_predictor

            hue = tree.predict_proba(dfs)[:, 1]
            tree_to_gating_max_path(
                tree,
                dfs,
                axes[:, idx],
                hue=hue,
                norm=norm,
                feature_names_in=feature_names,
                plot_rectangles=False,
                dirty_display_trick=True
            )

        for ax in axes.flatten():
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
        # Add a single color bar for all subplots
        if add_cbar:
            sm = plt.cm.ScalarMappable(cmap=attribution_palette, norm=norm)
            sm.set_array([])
            fig.subplots_adjust(right=0.85)  # Adjust right margin to fit the color bar
            cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('$\\hat{z}_{k}$', size=20, loc="center", labelpad=10, rotation=0)
        else:
            fig.tight_layout()


        plt.savefig(
            plots_path / f"fig-biplots_predict-{target_col}_ncells-{n_cells}",
            bbox_inches="tight"
        )
        plt.close()

        # plot tree themselves
        fig, axes = plt.subplots(len(disp_tubes), 1, figsize=(11, 4*len(disp_tubes)))
        for idx, tube in enumerate(disp_tubes):
            # In the edge case where we only want to display one tube,
            # then axes is not subscriptable
            if len(disp_tubes) == 1:
                ax = axes
            else:
                ax = axes[idx]
            tube_predictor = patient_predictor.tube_predictors_[idx]
            tree = tube_predictor.cell_predictor

            # Crazy hacky monkey patching
            exporter = _MPLTreeExporter(
                filled=True,
                feature_names=batch[0][idx].columns,
                impurity=False,
                label="all",
                fontsize=8,
                rounded=True,
            )
            exporter.get_fill_color = partial(
                custom_fill_color, threshold=tuner.best_threshold_
            )
            exporter.export(tree, ax=ax)
            ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]

            ax.text(
                -0.0,
                1.,
                f"Tube {tube}",
                transform=ax.transAxes,
                fontsize=20,
                fontweight='bold',
                va='top',
                ha='left'
            )
            fig.show()
        plt.savefig(f"plots/tree_target-{target_col}_tubes-{disp_tubes}.png", dpi=1000, bbox_inches="tight")
        plt.close()

