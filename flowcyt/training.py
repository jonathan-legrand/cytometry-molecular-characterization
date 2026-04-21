import time
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def print_training_params(X, y):
    dummy_preds = DummyClassifier(strategy="prior").fit(np.ones_like(X), y).predict(y)
    dummy_score = average_precision_score(y, dummy_preds)

    labels, counts = np.unique(y, return_counts=True)
    print("Class perc : ", counts / counts.sum()) 
    print("Dummy score : ", dummy_score) 


def parse_batch(batch, loader, return_id=False):
    """
    Batch unpacking must handle cases
    of dataset returning lots of stuff
    """
    if loader.dataset.return_patient_id:
        patient_id = batch.pop(-1)
    else:
        patient_id = None
    labels = batch[1]
    if loader.dataset.tabular_features is None:
        features = [batch[0]]
    else:
        features = [batch[0], batch[2]]
    if return_id:
        return features, labels, patient_id

    return features, labels


def train_epoch(model, loader, optimizer, loss_fn, device):
    training_loss = 0.
    data_time = AverageMeter("DataLoadingTime", ":6.3f")

    # Measure dataloading time
    end = time.time()
    for idx, batch in enumerate(loader):
        data_time.update(time.time() - end)

        features, labels = parse_batch(batch, loader)
        features = [f.to(device) for f in features]
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(*features).squeeze()
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        end = time.time()


    #print(data_time)
    return training_loss / (idx + 1) # Average across batches

# TODO Test that function?
def evaluate_epoch(model, val_loader, loss_fn, device):
    running_vloss = 0.0
    running_vacc = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            features, labels = parse_batch(vdata, val_loader)
            features = [f.to(device) for f in features]
            labels = labels.to(device)
            logits = model(*features).squeeze()

            vloss = loss_fn(logits, labels)
            probs = F.sigmoid(logits).cpu().numpy()

            running_vloss += vloss
            running_vacc += average_precision_score(labels.cpu(), probs)

    return running_vloss / (i+1), running_vacc / (i+1)
    