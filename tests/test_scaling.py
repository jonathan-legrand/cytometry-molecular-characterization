"""
Test sklearn's like Z-scoring functions.
"""
import numpy as np
import pytest
from flowcyt.scaling import CytoScaler

@pytest.fixture
def synthetic_data():
    # shape: (batch_size, n_panels, n_cells, n_markers)
    np.random.seed(42)
    X = np.random.randn(3, 2, 4, 5) * 2 + 5  # mean=5, std=2
    return X


def test_cytoscaler_transform_with_tabular_data(synthetic_data):
    tabular = np.random.randn(synthetic_data.shape[0], 5) * 3 + 7
    scaler = CytoScaler(include_tabular=True)
    scaler.fit(synthetic_data, tabular_data=tabular)
    X_scaled, tab_scaled = scaler.transform(synthetic_data, tabular_data=tabular)
    # Check cytometry data
    mean = X_scaled.mean(axis=(0,2))
    std = X_scaled.std(axis=(0,2))
    np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-7)
    np.testing.assert_allclose(std, np.ones_like(std), atol=1e-7)

    # Check tabular data
    np.testing.assert_allclose(
        tab_scaled.mean(axis=0), np.zeros(tabular.shape[1]), atol=1e-7
    )
    np.testing.assert_allclose(
        tab_scaled.std(axis=0), np.ones(tabular.shape[1]), atol=1e-7
    )

def test_cytoscaler_inverse_transform_roundtrip(synthetic_data):
    scaler = CytoScaler()
    scaler.fit(synthetic_data)
    X_scaled = scaler.transform(synthetic_data)
    X_recovered = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_recovered, synthetic_data, atol=1e-7)
