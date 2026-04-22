import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

from flowcyt.utils import name_to_dict
from flowcyt.results import lazy_load_folds
from flowcyt.evaluation import shorten, FoldResults
from sklearn.metrics import roc_auc_score
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.ticker as mticker
from itertools import combinations
from statannotations.Annotator import Annotator
import matplotlib as mpl

def extract_tags(name):
    dct = name_to_dict(name)
    return name.split("_")[0] + " (predict " + dct["predict"] + ")"

df = {}

if __name__ == "__main__":
    # Pos arg
    TARGET_COL = sys.argv[1]

    if TARGET_COL == "NPM":
        # For NPM
        expnames = [
            "age, sex, blast percentage_predict-NPM_stamp-20260421_155233",
            "finalrepo_predict-NPM_configname-huaspossiblenotab_nparams-726_stamp-20260421_155442",
            "finalrepo_predict-NPM_configname-huaspossible_nparams-738_stamp-20260421_151856",
            "cell-level-model_predict-NPM_ncells-10000_withtab-False_clf-decisiontreeclassifier_stamp-20260421_171507",
            "cell-level-model_predict-NPM_ncells-10000_withtab-True_clf-decisiontreeclassifier_stamp-20260421_175049"
        ]
        title = "$NPM1$-mutated AML prediction"
        letter = "A"
    elif TARGET_COL == "FLT3":
        # For FLT3
        expnames = [
            "age, sex, blast percentage_predict-FLT3_stamp-20250603_103945",
            "huaspossible_predict-FLT3_configname-huaspossiblenotab_nparams-726_stamp-20260108_181644",
            "huaspossible_predict-FLT3_configname-huaspossible_nparams-738_stamp-20260108_164152",
            "cell-level-model_predict-FLT3_ncells-10000_withtab-False_clf-decisiontreeclassifier_pooling-max_stamp-20260114_130040",
            "cell-level-model_predict-FLT3_ncells-10000_withtab-True_clf-decisiontreeclassifier_pooling-max_stamp-20260114_162408",
        ]
        title = "$FLT3$-ITD AML prediction"
        letter = "B"
    else:
        raise NotImplementedError()

    new_labels = [
        "Random forest\n(age, sex,\nblast percentage)",
        "CNN\n(Flow Cytometry)",
        "CNN\n(Flow Cytometry\n+clinical data)",
        "MIL decision\ntree\n(Flow Cytometry)",
        "MIL decision\ntree\n(Flow Cytometry\n+clinical data)",
    ]


    classifications_list = []
    for expname in expnames:
        exppath = Path("prediction") / expname
        fold_output = tuple(lazy_load_folds(exppath))
        scores = []
        scores_aug = []
        test_time_aug = False
        for output in fold_output:
            scores.append(roc_auc_score(*output[:2]))
            try:
                fold_df = pd.DataFrame(output).T.rename(lambda x: ["y_true", "y_score", "Patient"][x], axis=1)
            except IndexError:
                output_short = (*output[:2], output[-1])
                fold_df = pd.DataFrame(output_short).T.rename(lambda x: ["y_true", "y_score", "Patient"][x], axis=1)

            fold_df["expname"] = expname
            classifications_list.append(fold_df)

        df[expname] = tuple(scores)

    df = pd.DataFrame(df)

    df["fold"] = list(range(10))

    classifications = pd.concat(classifications_list)

    # Check that experiments were split in exactly the same way
    ref_exp = classifications.loc[classifications.expname == expnames[0]]
    for expname in expnames:
        comp_exp = classifications[classifications.expname == expname]
        assert np.all(ref_exp.Patient == comp_exp.Patient)

    classifications["y_pred"] = (classifications.y_score > 0.5)
    classifications["correctly_classified"] = (classifications.y_pred == classifications.y_true).astype(bool)
    classifications["short_name"] = classifications.expname.apply(extract_tags)

    sns.displot(
        classifications,
        x="y_score",
        hue="y_true",
        row="expname",
        facet_kws=dict(xlim=(0, 1))
    )
    plt.savefig(f"plots/predict-{TARGET_COL}_preds-distribution.png")
    plt.close()

    results_table = classifications.pivot_table(columns="expname" ,values="correctly_classified", index="Patient")

    cmap = LinearSegmentedColormap.from_list('Custom', ("tab:red", "tab:green"), 2)
    norm = BoundaryNorm(boundaries=[0, 1], ncolors=2)

    plt.subplots(figsize=(20, 3))
    g = sns.heatmap(
        results_table.T,
        cmap=cmap,
        cbar_kws=dict(cmap=cmap, norm=norm, ticks=[0, 1], format=mticker.FixedFormatter(["Bad classification", "Correct classification"])),
        linewidths=0.1
    )
    plt.savefig(f"plots/predict-{TARGET_COL}_bad-vs-good.png")
    plt.close()

    preds_table = classifications.pivot_table(columns="short_name" ,values="y_score", index="Patient").astype(float)
    labels = classifications.pivot_table(columns="short_name" ,values="y_true", index="Patient").astype(int)

    m = df.melt(id_vars="fold", value_name="roc_auc", var_name="run_name")
    m["short_name"] = m["run_name"].apply(extract_tags)
    m["hue_col"] = m["fold"].astype(str) + m["short_name"]

    plt.subplots(figsize=(10, 6))
    g = sns.barplot(
        m,
        x="fold",
        y="roc_auc",
        hue="short_name",
        palette="Paired"
    )
    handles, labels = g.get_legend_handles_labels()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.ylim(0, 1)
    plt.savefig(f"plots/predict-{TARGET_COL}_score-per-fold.png")
    plt.close()

    # Permutation testing
    os.makedirs("plots/permtest", exist_ok=True)
    def plot_perm_test(null_distribution, realised_stat):
        n_perms = len(null_distribution)
        f, ax = plt.subplots(figsize=(10, 5))
        h = ax.hist(null_distribution, histtype="step", label=f"Null statistics, {n_perms} permutations")
        ax.vlines(realised_stat, ymin=0, ymax=np.max(h[0]), color="red", label="Realised difference")
        ax.legend(loc="upper left")
        ax.set_xlabel("ROC-AUC")

    rng = np.random.default_rng(seed=1234)
    names = m.run_name.unique()
    pairs = tuple(combinations(names, 2))
    N_PERMS = 1000
    n_folds = df["fold"].max()
    indices = np.arange(n_folds * 2)

    def stat_func(sample_a, sample_b):
        return np.median(sample_a - sample_b)

    test_results = []

    for pair in pairs:
        if pair[0] == pair[1]:
            continue
        perfs_a = df[pair[0]].values
        perfs_b = df[pair[1]].values
        realised_stat = stat_func(perfs_a, perfs_b)
        all_perfs = df.loc[:, pair].values.flatten()
        null_distribution = np.zeros(N_PERMS)
        for idx in range(N_PERMS):
            swapped = rng.integers(0, 2, size=(n_folds+1))
            perm_a = np.where(swapped, perfs_b, perfs_a)
            perm_b = np.where(swapped, perfs_a, perfs_b)
            null_distribution[idx] = stat_func(perm_a, perm_b)
            
        p_more = (realised_stat <= null_distribution).sum() / N_PERMS
        p_less = (realised_stat >= null_distribution).sum() / N_PERMS
        p_value = min(p_more, p_less) * 2

        if p_value < 0.1:
            plot_perm_test(null_distribution, realised_stat)
            plt.title(f"{pair[0]} - {pair[1]}\np-value = {p_value}")
            plt.savefig(f"plots/permtest/predict-{TARGET_COL}_{pair}.png")
            plt.close()

        test_results.append(
            dict(classifier_a=pair[0], classifier_b=pair[1], statistic=realised_stat, p_value=p_value)
        )

    test_results = pd.DataFrame(test_results)

    # We want linear increase in brightness for black and white printing
    # have different hues, and have the darkest color not too dark 
    # so as to still see the swarmplot
    palette = sns.cubehelix_palette(
        n_colors=3, start=2, rot=1, dark=0.5, light=0.9
    )

    f, ax = plt.subplots(figsize=(11, 5))
    order = m["run_name"].unique()
    sns.boxplot(
        m, 
        x="run_name",
        y="roc_auc",
        hue="short_name",
        palette=palette,
        ax=ax,
        order=order,
        fliersize=0,
        legend=False,
    )
    sns.swarmplot(
        m,
        x="run_name",
        y="roc_auc",
        ax=ax,
        order=order,
        zorder=2,
        legend=None,
        color="black"
    )

    plt.grid()
    ax.set_title(
        title,
        size=15
    )
    ax.set_xticks(range(len(new_labels)))

    ax.set_xticklabels(
        new_labels,
        size=13
    )

    ax.set_xlabel(None)
    ax.set_ylabel("Model Performance [AUROC]", size=13)

    annot = Annotator(
        ax,
        pairs,
        data=m,
        x="run_name",
        y="roc_auc",
        order=order
    )

    (annot
     .configure(
         test=None,
         test_short_name=f"Permutation test {N_PERMS}",
         hide_non_significant=True,
         show_test_name=True
    )
     .set_pvalues(pvalues=test_results.p_value)
     .annotate()
    )
    ax.text(
        -0.05,
        1.1,
        letter,
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='left'
    )

    ax.set_yticks([i/5 for i in range(3,6)])
    plt.savefig(f"plots/{title}.svg", dpi=1000)
    plt.close()
