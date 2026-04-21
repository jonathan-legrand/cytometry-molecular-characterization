
import warnings
import copy

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import copy

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator

from flowcyt.results import tensor_to_df
from flowcyt.inference import confusion_mapping
from flowcyt.training import parse_batch

rng = np.random.default_rng(seed=1234)


from scipy.stats import gaussian_kde
from scipy.signal import argrelmax

class Discretizer(BaseEstimator):
    def __init__(
            self,
            bw=0.10,
            n_values=1000,
            relmaxorder=55,
            min_mode_density=1e-2,
            n_modes=3
        ):
        self.bw = bw
        self.n_values = n_values
        self.relmaxorder = relmaxorder
        self.min_mode_density = min_mode_density
        self.n_modes = n_modes
    
    def fit(self, X, y=None):
        
        self.f_mapping_ = {}
        for marker in X.columns:
            vals_raw = X.loc[:, marker].values
            vals = (vals_raw - vals_raw.mean()) / vals_raw.std()
            x = np.linspace(vals.min(), vals.max(), num=self.n_values)
            self.vals = vals
            self.f_mapping_[marker] = dict()
            f_ = gaussian_kde(vals, bw_method=self.bw)
            
            modes_args = argrelmax(f_(x), order=self.relmaxorder)[0]
            modes = x[modes_args]
            
            # Topk modes selection, it is so bad
            mode_density = f_(x)[modes_args]
            try:
                select_idx = np.argsort(mode_density)[::-1][:self.n_modes]
                modes = modes[select_idx]
                modes_args = modes_args[select_idx]
            except IndexError:
                # Remove modes that contain few cells
                mode_mask = mode_density > self.min_mode_density
                modes = modes[mode_mask]
                modes_args = modes_args[mode_mask]


            # Is it worth keeping track of so many things?
            self.f_mapping_[marker]["f"] = f_
            self.f_mapping_[marker]["modes"] = modes
            self.f_mapping_[marker]["modes_args"] = modes_args
            self.f_mapping_[marker]["x"] = x
            self.f_mapping_[marker]["vals"] = vals

        return self

    def transform(self, X, y=None):
        discrete_X = []
        for marker in X.columns:
            modes = self.f_mapping_[marker]["modes"]
            vals_raw = X.loc[:, marker].values
            vals = (vals_raw - vals_raw.mean()) / vals_raw.std()

            distances_to_modes = np.abs(vals.reshape((-1, 1)) - modes.reshape((1, len(modes))))
            discretized_vals = np.argmin(distances_to_modes, axis=1)
        
            discrete_col = pd.DataFrame(discretized_vals, columns=[marker])
            discrete_X.append(discrete_col)

        discrete_X = pd.concat(discrete_X, axis=1)
        return discrete_X




def prune_majority_bin(X, y, additional_array=None, pruning_ratio=0.9):
    edges = np.histogram_bin_edges(y, bins=6)
    y_digits = np.digitize(y, edges)
    indices, counts = np.unique(y_digits, return_counts=True)
    max_idx = np.argmax(counts)
    over_represented_bin = indices[max_idx]
    over_represented_idx = np.where(y_digits == over_represented_bin)[0]
    remove_idx = rng.choice(over_represented_idx, size=int(pruning_ratio * counts[max_idx]), replace=False)
    print(indices, counts)
    
    if additional_array:
        return X.drop(index=remove_idx), np.delete(y, remove_idx), np.delete(additional_array, remove_idx)

    return X.drop(index=remove_idx), np.delete(y, remove_idx)


def plot_patient_attribution_norm(
    cells,
    attr_df,
    epsilon=1e-6,
    **scatter_kwargs
    ):
    """
    Adding epsilon avoids numerical errors when computing log scale for size
    """
    
    # L1 norm
    attr_norm = attr_df.drop("tube", axis=1).apply(np.abs).apply(np.sum, axis=1)
    attr_norm += epsilon
    
    # TODO Combiner avec l'importance de SS?
    for marker in "FS-A", "CD45 KO":
        print(marker)
        g = sns.relplot(
            cells[~cells[marker].isna()],
            col="tube",
            x=marker,
            y="SS-A",
            legend=True,
            hue=attr_norm,
            **scatter_kwargs
        )
        yield g

def plot_patient_attribution(
    cells,
    attr_df,
    markers_of_interest=None,
    ref_marker="SS-A",
    biplot_hue=False,
    **scatter_kwargs
    ):
    
    if markers_of_interest is None:
        markers_of_interest = attr_df.drop("tube", axis=1).abs().max(axis=0).sort_values(ascending=False).index
    
    for marker in markers_of_interest:
        print(ref_marker, marker)
        
        # Select attr for cells which are represented
        # in both dimensions
        marker_msk = ~attr_df[marker].isna()
        ref_msk = ~attr_df[ref_marker].isna()
        msk = marker_msk & ref_msk
        if msk.sum() == 0:
            warnings.warn(
                f"{ref_marker} and {marker} are not in the same tube, default to SS-A")
            ref_marker = "SS-A"
            markers_of_interest.append(ref_marker)
            ref_msk = ~attr_df[ref_marker].isna()
            msk = marker_msk & ref_msk
            
        if biplot_hue:
            ig_marker = attr_df[msk][marker].dropna().to_numpy()
            ig_ref = attr_df[msk][ref_marker].dropna().to_numpy() # In case marker has not 3 tubes
            ig_sum = ig_marker + ig_ref
        else:
            ig_sum = attr_df[msk].select_dtypes(include=float).sum(axis=1)

        ig_norm = attr_df[msk].select_dtypes(include=float).pow(2).sum(axis=1).pow(1/2)

        cells_disp = cells[~cells[marker].isna()].copy()
        cells_disp["ig_norm"] = ig_norm
        cells_disp["ig_sum"] = ig_sum
        g = sns.relplot(
            cells_disp,
            col="tube",
            x=marker,
            y=ref_marker,
            legend="auto",
            size="ig_norm",
            hue="ig_sum",
            **scatter_kwargs
        )
        yield g

tabular_features_one_hot = ['Blastes moelle osseuse (%)', 'age', 'is_male', 'is_female']
       

tube_to_idx = {
    "A": 0,
    "B": 1,
    "C": 2
}

def plot_attention(
        model,
        dset,
        scaler,
        tube="A",
        thresh=0.5,
        **scatter_kwargs
    ):

    dl = DataLoader(dset, batch_size=1)
    meta = dset.metadata
    target_col = dset.target_col
    for f, l, r in dl:
        
        with torch.no_grad():
            logit, A = model.pred_with_attn(f)
        prob = float(F.sigmoid(logit))
        decision = "mutated" if prob > thresh else "WT"

        f = scaler.inverse_transform(f.detach())
        sample_arr = f.squeeze().numpy()
        
        A = A.detach().numpy().flatten()

        tube_idx = tube_to_idx[tube]
        patient_row = meta.loc[meta.ID == r[tube_idx], :].squeeze()
        patient_key = patient_row.Patient

        sample = dset.fc_dataset[patient_key][tube]
        pns_labels = copy.copy(sample.pns_labels)
        for bad_label in "TIME", "FS-H", "":
            try:
                pns_labels.remove(bad_label)
            except ValueError:
                continue
        
        df = pd.DataFrame(sample_arr[tube_idx, ...], columns=pns_labels)
        df["attention"] = A
        df["is_important"] = np.where(A > 1/5000, 1, 0)

        if l == 1:
            if decision == "mutated":
                title_color = "green" # TP
            else:
                title_color = "red" # FN
        else:
            if decision == "mutated":
                title_color = "red" # FP
            else:
                title_color = "green" # TN

        fig, axes = plt.subplots(
            1, 4, sharey=True, figsize=(15, 5)
        )
        
        sns.scatterplot(
            df,
            x="CD33 PC5.5",
            y="SS-A",
            ax=axes[0],
            legend=False,
            **scatter_kwargs
        )
        axes[0].set_ylim(5, 13)
        axes[0].set_xlim(-7.5, 11)
        
        sns.scatterplot(
            df,
            x="CD34 PC7",
            y="SS-A",
            ax=axes[1],
            legend=False,
            **scatter_kwargs
        )
        axes[1].set_xlim(-7.5, 11)

        sns.scatterplot(
            df,
            x="FS-A",
            y="SS-A",
            ax=axes[2],
            legend=False,
            **scatter_kwargs
        )
        sns.scatterplot(
            df,
            x="CD45 KO",
            y="SS-A",
            ax=axes[3],
            legend="brief",
            **scatter_kwargs
        )
        axes[3].set_xlim(0, 12)

        fig.suptitle(
            f"{patient_row.Patient} : {target_col} {patient_row[target_col]}, predicted {decision} ({prob})",
            color=title_color
        )
        yield fig, axes

def plot_by_outcome(summary, cmap, cbar_label, symlognorm=False):
    num_summary = summary.select_dtypes(include=float)
    vmin = num_summary.min().min()
    vmax = num_summary.max().max()
    
    # That's were the trouble begins
    # We want a linear colorscale for most values, but the
    # outliers are crushing the scale. So we set a linear
    # range for 99% of the values and send the rest to 
    # the logarithmic world
    if symlognorm: 
        linthresh = np.quantile(np.abs(num_summary), 0.95)
        # Define percentile boundaries
        norm = mcolors.SymLogNorm(
            linthresh=linthresh,
            linscale=10,
            vmin=vmin,
            vmax=vmax,
            base=10
        )
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
    heatkwargs = dict(
        cmap=cmap,
        norm=norm,
        cbar_kws={"label": cbar_label},
    )

    for outcome in summary.Outcome.unique():

        num_attr = summary.select_dtypes(include=float)
        relative_marker_importance = num_attr
        relative_marker_importance["Patient"] = summary.Patient

        mask = summary.Outcome == outcome
        melt = relative_marker_importance[mask].melt(
            id_vars=["Patient"], var_name="Marker", value_name="Attribution"
        )

        p = melt.pivot(columns='Patient', index="Marker")
        title_color = "green" if outcome in {"TN", "TP"} else "red"
        f, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(p.sort_index(), ax=ax,  **heatkwargs)
        title = f"Local interpretation for {outcome} patients"
        plt.suptitle(title, color=title_color, y=0.95)
        plt.xlabel(None)
        plt.ylabel(None)
        plt.show()


def clustermap_by_outcome(summary, cmap, cbar_label, symlognorm=False):
    num_summary = summary.select_dtypes(include=float)
    vmin = num_summary.min().min()
    vmax = num_summary.max().max()
    
    # That's were the trouble begins
    # We want a linear colorscale for most values, but the
    # outliers are crushing the scale. So we set a linear
    # range for 99% of the values and send the rest to 
    # the logarithmic world
    if symlognorm: 
        linthresh = np.quantile(np.abs(num_summary), 0.95)
        # Define percentile boundaries
        norm = mcolors.SymLogNorm(
            linthresh=linthresh,
            linscale=10,
            vmin=vmin,
            vmax=vmax,
            base=10
        )
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
    heatkwargs = dict(
        cmap=cmap,
        norm=norm,
        cbar_kws={"label": cbar_label},
    )

    for outcome in summary.Outcome.unique():

        num_attr = summary.select_dtypes(include=float)
        relative_marker_importance = num_attr
        relative_marker_importance["Patient"] = summary.Patient

        mask = summary.Outcome == outcome
        melt = relative_marker_importance[mask].melt(
            id_vars=["Patient"], var_name="Marker", value_name="Attribution"
        )

        p = melt.pivot(columns='Patient', index="Marker")
        title_color = "green" if outcome in {"TN", "TP"} else "red"
        sns.clustermap(
            p.sort_index(), 
            col_cluster=False,
            **heatkwargs
        )
        plt.suptitle(outcome, color=title_color)
        plt.show()
