import torch
from flowcyt.cnn import AttentionCytoNet

from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import _utils

from flowcyt.training import train_epoch

def test_train_epoch_loss_decreases(dataset, model_wrapper):
    # Simple linear model and dataset
    loader = DataLoader(dataset, batch_size=10)
    optimizer = torch.optim.SGD(model_wrapper.model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    model_wrapper.model.train()
    train_epoch(model_wrapper, loader, optimizer, loss_fn, device="cpu")
