import numpy as np
import torch
import torch.nn.functional as F


QUANTILE_GENERATOR = [i/100 for i in range(0, 100)]
def remove_attr_values(model, f, attr, quantile_gen=QUANTILE_GENERATOR):
    """
    Remove cells which have the least attribution,
    by using thresholds which correspond to a sequence of quantiles
    """
    model_outputs = []
    initial_response = float(F.sigmoid(model(f)).detach())
    model_outputs.append((0, initial_response))

    for quantile_prob in quantile_gen:
        abs_arr = attr.abs()
        q_abs = np.quantile(abs_arr, quantile_prob, method="closest_observation")
        masked_f = torch.where(abs_arr < q_abs, 0, f)
        model_output = F.sigmoid(model(masked_f))
        model_outputs.append(
            (
                quantile_prob, float(model_output)
            )
        )
    
    return model_outputs


def remove_cells(model, f, attr, quantile_gen=QUANTILE_GENERATOR):
    model_outputs = []
    initial_response = float(F.sigmoid(model(f)).detach())
    model_outputs.append((0, initial_response))
    n_cells = f.shape[2]
    
    for quantile_prob in quantile_gen:
        cell_mean_attr = attr.squeeze().abs().mean(axis=1)
        q_abs = np.quantile(cell_mean_attr, quantile_prob, method="closest_observation")
    
        # Expand the mask and filter with element-wise multiplication
        # There might be a smarter way to do this
        mask = torch.where(cell_mean_attr < q_abs, 0, 1)
        expanded_mask = torch.stack([mask for _ in range(n_cells)], axis=1)
    
        masked_f = f * expanded_mask
        model_output = F.sigmoid(model(masked_f))
        model_outputs.append(
            (
                quantile_prob, float(model_output)
            )
        )
    return model_outputs