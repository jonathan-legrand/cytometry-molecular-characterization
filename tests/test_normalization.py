"""
Test normalization layers for neural networks
"""
import pytest

import torch
from torch.nn import BatchNorm2d

BATCH_SIZE = 32
K = 5000
H = 6
BATCH_SIZE = 32

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(1234)

def test_permutation_invariance():
    X = torch.rand((BATCH_SIZE, H, K, 1))
    perm = torch.randperm(K)
    X_perm = X[:, :, perm, :]

    normalizer = BatchNorm2d(H, 1)
    Y = normalizer(X)
    Y_perm = normalizer(X_perm)
    Y_perm_back = Y_perm[:, :, torch.argsort(perm), :]
    torch.testing.assert_close(Y_perm_back, Y)

def test_normalized():
    X = torch.rand((BATCH_SIZE, H, K, 1))
    normalizer = BatchNorm2d(H, 1)
    Y = normalizer(X)
    Y_mean = Y.mean(axis=(0, 2))
    Y_std = Y.std(axis=(0, 2))
    zeros = torch.zeros_like(Y_mean)
    ones = torch.ones_like(Y_mean)

    torch.testing.assert_close(Y_mean, zeros)
    torch.testing.assert_close(
        Y_std,
        ones,
        atol=normalizer.eps,
        rtol=1e-3
    )

