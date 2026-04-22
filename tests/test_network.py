"""
Testing functions for the CNNs
"""
from flowcyt.cnn import AttentionCytoNet, AttentionNetWithMetadata
import pytest
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 


@pytest.fixture(autouse=True)
def seeding():
    torch.manual_seed(2025)

@pytest.fixture
def random_features(run_config):
    model_spec = run_config["model_spec"]
    batch_size = run_config["batch_size"]
    features_shape = (
        batch_size,
        model_spec["n_tubes"],
        model_spec["n_cells"],
        model_spec["n_markers"],
    )
    features = torch.rand(features_shape)
    return features


@pytest.fixture
def tabular_features(run_config): # TODO Read tabular from run_config?
    batch_size = run_config["batch_size"]
    tabular_features = run_config.get("tabular_features", None)
    if tabular_features is None:
        return None

    n_tabular_features = len(tabular_features)
    if "sexe" in tabular_features:
        n_tabular_features += 1 # Sex is one hot encoded
    tabular_X = torch.rand((batch_size, n_tabular_features))
    return tabular_X

@pytest.fixture
def one_features(run_config):
    model_spec = run_config["model_spec"]
    batch_size = run_config["batch_size"]
    features_shape = (
        batch_size,
        model_spec["n_tubes"],
        model_spec["n_cells"],
        model_spec["n_markers"],
    )
    features = torch.ones(features_shape)
    return features


@torch.no_grad()
def test_output_shape(model_wrapper, random_features, tabular_features, run_config):
    batch_size = run_config["batch_size"]
    outputs = model_wrapper(random_features, tabular_features)
    assert outputs.shape == (batch_size, 1)



@torch.no_grad()
def test_no_nans_or_infs(model_wrapper, random_features, tabular_features):
    y = model_wrapper(random_features, tabular_features)
    assert not torch.isnan(y).any(), "Output contains NaNs"
    assert not torch.isinf(y).any(), "Output contains Infs"

@torch.no_grad()
def test_attention_computation(model_wrapper, run_config):
    model = model_wrapper.model
    B, K, M = 1, run_config["model_spec"]["n_cells"], model.M
    identical_H = torch.randn(1, M).expand(B, K, M)

    A = model._compute_attn(identical_H, softmax=True)

    expected_attn = torch.full_like(A, 1 / K)
    assert torch.allclose(A, expected_attn, atol=1e-6), "Attention not uniform"


@pytest.fixture()
def model(model_wrapper):
    """
    Fixture for tests which only need access to the model
    and won't call it directly
    """
    return model_wrapper.model


@torch.no_grad()
def test_consistent_embedding(model, one_features):
    H = model.conv1(one_features)
    for i in range(H.shape[2]-1):
        torch.testing.assert_close(
            H[:, :, i, :], H[:, :, i+1, :]
        ), "Identical cells do not receive identical embeddings"

@torch.no_grad()
def test_attention_permutation_equivariance(model, random_features):
    """
    The whole model should be permutation-invariant after pooling.
    Before pooling, it should be permutation-equivariant.
    """
    X = random_features
    B, C, K, D = X.shape
    perm = torch.randperm(K)
    X_perm = X[:, :, perm, :]

    H_1 = F.relu(model.conv1(X))
    H = model._convolve(X)
    A = model._compute_attn(H, softmax=False)

    H_1_perm = F.relu(model.conv1(X_perm))
    H_perm = model._convolve(X_perm)
    A_perm = model._compute_attn(H_perm, softmax=False)

    H_1_perm_back = H_1_perm[:, :, torch.argsort(perm), :]
    # Reshaping black magic that becomes clearer if you take a look at _convolve method
    H_perm_back = H_perm.reshape((B, C, K, -1))[:, :, torch.argsort(perm), :].reshape((B, C*K, -1))
    A_perm_back = A_perm.reshape((B, C, K))[:, :, torch.argsort(perm)].reshape((B, 1, -1))

    torch.testing.assert_close(H_1, H_1_perm_back, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(H, H_perm_back, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(A, A_perm_back, atol=1e-5, rtol=1e-3)


@torch.no_grad()
def test_equal_attention(model, one_features, run_config):
    """
    Test that identical cells get the same attention (per tube)
    """
    model_spec = run_config["model_spec"]
    C = model_spec["n_tubes"]
    K = model_spec["n_cells"]

    H = model._convolve(one_features)
    
    # Softmax introduces huge differences from
    # tiny numerical differences in the logits,
    # so we round logits in a very brutal way
    #  to get consistent attentions.
    H = H.round(decimals=6)
    A = model._compute_attn(H)
    
    # Identical cells must receive identical attention,
    # however cells between tubes cannot be identical since
    # markers are different. So we need to split verification
    # by tube of origin
    for idx in range(C):
        tube_attention = A[0, 0, idx*K:(idx+1) * K] 
        expected_attn = torch.full_like(tube_attention, tube_attention[0])
    
        assert torch.allclose(
            tube_attention,
            expected_attn,
            rtol=1e-3,
            atol=1e-6
        ), f"Attention not uniform: max diff = {(A - expected_attn).abs().max()}"

@torch.no_grad()
def test_permutation_invariance(
        model_wrapper,
        random_features,
        run_config,
        tabular_features
    ):
    """
    Our model should be invariant to cell permutation
    """
    n_cells = run_config["model_spec"]["n_cells"]
    perm_idx = torch.randperm(n_cells)
    permuted_features = random_features[:, :, perm_idx, :]
    preds = model_wrapper(random_features, tabular_features)
    permuted_preds = model_wrapper(permuted_features, tabular_features)
    assert torch.all(
        torch.isclose(
            preds,
            permuted_preds,
        )
    )