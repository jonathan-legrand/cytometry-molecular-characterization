import numpy as np
import torch

from flowcyt.results import df_to_tensor
import pandas as pd

rng = np.random.default_rng(seed=1234)

def compute_downsampled(target_cell, df, model, tabular, cytometry_shape, rng=rng):
    """
    Compute effect of removing a target cell (replacing it with another random cell)
    """
    
    target_tube = target_cell.tube
    tube_df = df[df.tube == target_tube].copy()
    
    # Get target cell index in the tube
    target_idx = target_cell.name
    
    # Get other cell indices in the same tube
    other_indices = tube_df[tube_df.index != target_idx].index.tolist()
    
    # Choose replacement cells
    replace_with_idx = rng.choice(other_indices, size=1)[0]
    
    # Create modified DataFrame
    modified_df = df.copy()
    
    # Replace target cell with a random other cell from same tube
    modified_df.loc[target_idx] = tube_df.loc[replace_with_idx]
    
    # Convert to tensor
    modified_input = df_to_tensor(modified_df).reshape(cytometry_shape)
    
    with torch.no_grad():
        modified_output, modified_attn = model.pred_with_attn(modified_input, tabular)
    
    return modified_output, modified_attn

def compute_upsampled(
        target_cell,
        df,
        model,
        tabular,
        cytometry_shape,
        n_replacements=1,
        random_replacement=True,
        rng=rng
    ):
    """
    If random replacement is False, replace cells that have
    the least attention so that delta_y is mostly
    influenced by upsampled cell rather than replacement
    """
    target_tube = target_cell.tube
    resampled_input = df.copy()

    tube_idx = resampled_input[resampled_input.tube == target_tube].index
    
    if random_replacement:
        replace_idx = rng.choice(
            resampled_input.iloc[tube_idx].index,
            size=n_replacements,
            replace=False
        )
    else:
        original_input = df_to_tensor(df).reshape(cytometry_shape)
        _, attention = model.pred_with_attn(original_input, tabular)
        resampled_input["attention"] = attention.detach().numpy()
        replace_idx = resampled_input.iloc[tube_idx, :].sort_values(by="attention")[:n_replacements].index
        resampled_input.drop("attention", axis=1, inplace=True)

    # Inject noise to target cell when upsampling
    target_marker_values = target_cell.drop("tube")
    target_marker_values = pd.concat([target_marker_values]*n_replacements, axis=1).T.reset_index(drop=True)
    target_marker_values += rng.normal(scale=0.01, size=target_marker_values.shape)
    target_marker_values = target_marker_values.astype(np.float32)
    target_marker_values["tube"] = target_tube
    target_marker_values.index = replace_idx
    resampled_input.update(target_marker_values)
    modified_input = df_to_tensor(resampled_input).reshape(cytometry_shape)
    with torch.no_grad():
        modified_output, modified_attn = model.pred_with_attn(modified_input, tabular)
    return modified_output, modified_attn
