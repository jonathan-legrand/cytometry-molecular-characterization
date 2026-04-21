import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np

class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        assert min_delta >= 0
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def check(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False
        elif current_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False

    def log(self):
        print(f"""Loss has not improved for {self.patience}epochs (best val = {self.best_loss})
              Break training loop. """
        )
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pool_out_dim(h_in, w_in, p):
    h_ker, w_ker = p.kernel_size
    try:
        h_out = ((h_in + 2*p.padding - p.dilation * (h_ker - 1) - 1) )/ p.stride[0] + 1
        w_out = ((w_in + 2*p.padding - p.dilation * (w_ker - 1) - 1) )/ p.stride[1] + 1
    except AttributeError: # Oh no average pooling
        h_out = ((h_in + 2*p.padding - 1 * (h_ker - 1) - 1) )/ p.stride[0] + 1
        w_out = ((w_in + 2*p.padding - 1 * (w_ker - 1) - 1) )/ p.stride[1] + 1

    assert h_out == int(h_out)
    assert w_out == int(w_out)
    return int(h_out), int(w_out)



class AttentionCytoNet(nn.Module):
    def __init__(
            self,
            n_tubes=3,
            n_cells=10000,
            n_markers=12,
            n_hidden_blocks=2,
            hidden_channels=8,
            hidden_layers=2,
            hidden_dense_units=9,
            dropout=0,
            L=100, 
            *args,
            **kwargs
        ):
        super().__init__()

        self.L = L
        self.M = hidden_channels // n_tubes
        self.C = n_tubes
        self.hidden_layers = hidden_layers
        self.hidden_dense_units = hidden_dense_units
        self.dropout = dropout
        
        # Feature extraction
        self.conv1 = nn.Conv2d(
            in_channels=n_tubes,
            out_channels=hidden_channels,
            kernel_size=(1, n_markers),
            groups=n_tubes, # Not the same cells so convolve tubes separatly
            bias=False
        )
        
        pointwise_convs = nn.ModuleList()
        for _ in range(n_hidden_blocks):
            pointwise_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(1, 1),
                    bias=False, # Bias useless before BN
                    groups=n_tubes
                ),
                nn.BatchNorm2d(
                    num_features=hidden_channels,
                )
            )
            pointwise_convs.append(pointwise_block)
        self.pointwise_convs = pointwise_convs

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # One attention value per cell
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False)
        )
        self._setup_dense()
    
    def _setup_dense(self):
        dense_1 = nn.Linear(
            in_features=self.M,
            out_features=self.hidden_dense_units
        )
        dense_layers = nn.ModuleList([dense_1])
        for _ in range(self.hidden_layers):
            dense_layers.append(
                nn.Sequential(nn.Linear(
                    in_features=self.hidden_dense_units,
                    out_features=self.hidden_dense_units
                ))
            )
        self.dense_layers = dense_layers
        self.classifier = nn.Linear(
                in_features=self.hidden_dense_units,
                out_features=1
        )
        
    def _compute_attn(self, H, softmax=True):
        """
        H should be of shape (B, C * K, M)
        returns A of shape (B, K)
        """
        A = self.attention(H)
        if softmax is True:
            A = F.softmax(A, dim=1) # Softmax over K
        A = A.permute((0, 2, 1)) # For matrix multiplication
        return A

    
    def _convolve(self, X):
        B, E, K, D = X.shape # E = C*M
        H = self.conv1(X)
        H = F.relu(H)
        for layer in self.pointwise_convs:
            H = layer(H)
            H = F.relu(H)

        # Remove marker dim of size 1
        H = H.squeeze(-1) # (B, C*M, K)
        H = H.view(B, self.C, self.M, K) # (B, C, M, K)
        H = H.permute(0, 1, 3, 2) # (B, C, K, M)
        H = H.reshape(B, self.C*K, self.M) # (B, C*K, M)
        return H

    def forward(self, X):
        y_pred, _ = self.pred_with_attn(X)
        return y_pred

    def pred_with_attn(self, X, tabular=None):
        H = self._convolve(X)
        A = self._compute_attn(H)
        Z = torch.bmm(A, H) # (B, M)
        Z = Z.squeeze() # Ok that would not work for more attention branches

        for layer in self.dense_layers:
            Z = layer(Z)
            Z = F.relu(Z)
            Z = F.dropout(Z, p=self.dropout)

        y_pred = self.classifier(Z)
        return y_pred, A

class AttentionNetWithMetadata(AttentionCytoNet):
    def __init__(
            self,
            n_tubes=3,
            n_cells=10000,
            n_markers=12,
            n_hidden_blocks=2,
            hidden_channels=8,
            hidden_layers=2,
            hidden_dense_units=9,
            n_tabular_variables=4,
            dropout=0,
            L=100,
            *args,
            **kwargs
        ):
        self.n_tabular_variables = n_tabular_variables
        super().__init__(
            n_tubes,
            n_cells,
            n_markers,
            n_hidden_blocks,
            hidden_channels,
            hidden_layers,
            hidden_dense_units,
            dropout,
            L,
            *args,
            **kwargs
        )
    
    def _setup_dense(self):
        n_units_in = self.M + self.n_tabular_variables
        dense_1 = nn.Linear(
            in_features=n_units_in,
            out_features=self.hidden_dense_units
        )
        dense_layers = nn.ModuleList([dense_1])
        for _ in range(self.hidden_layers):
            dense_layers.append(
                nn.Sequential(nn.Linear(
                    in_features=self.hidden_dense_units,
                    out_features=self.hidden_dense_units
                ))
            )
        self.dense_layers = dense_layers
        self.classifier = nn.Linear(
                in_features=self.hidden_dense_units,
                out_features=1
        )
       

    def pred_with_attn(self, X, tabular_data):
        H = self._convolve(X)
        A = self._compute_attn(H)
        Z = torch.bmm(A, H)
        Z = Z.squeeze()

        if Z.ndim == 1:
            Z = Z.reshape((1, *Z.shape))
        Z = torch.cat((Z, tabular_data), dim=1)

        for layer in self.dense_layers:
            Z = layer(Z)
            Z = F.relu(Z)
            Z = F.dropout(Z, p=self.dropout)


        y_pred = self.classifier(Z)
        return y_pred, A.squeeze()
    
    def forward(self, X, tabular_data): # Or overload?
        y_pred, _ = self.pred_with_attn(X, tabular_data)
        return y_pred

class CytoNetWithMetadata(AttentionNetWithMetadata):

    def pred_with_attn(self, X, tabular_data):
        H = self._convolve(X)
        Z = F.avg_pool2d(H, kernel_size=(H.shape[1], 1))
        Z = Z.squeeze()

        if Z.ndim == 1:
            Z = Z.reshape((1, *Z.shape))
        Z = torch.cat((Z, tabular_data), dim=1)

        for layer in self.dense_layers:
            Z = layer(Z)
            Z = F.relu(Z)
            Z = F.dropout(Z, p=self.dropout)


        y_pred = self.classifier(Z)
        return y_pred, None

class CytoNet(AttentionCytoNet):
    def pred_with_attn(self, X, tabular=None):
        H = self._convolve(X)
        Z = F.avg_pool2d(H, kernel_size=(H.shape[1], 1))
        Z = Z.squeeze()

        for layer in self.dense_layers:
            Z = layer(Z)
            Z = F.relu(Z)
            Z = F.dropout(Z, p=self.dropout)

        y_pred = self.classifier(Z)
        return y_pred, None


class ModelWrapper:
    """
    We use a wrapper for init and forward, to gracefully handle
    tabular features.
    """
    def __init__(self, tabular_features=None, pooling="attention", **model_spec):
        
        if tabular_features is None:
            if pooling == "attention":
                self.model = AttentionCytoNet(**model_spec)
            else:
                self.model = CytoNet(**model_spec)
            
            self.model.n_tabular_variables = 0
        else:
            n_tabular_features = len(tabular_features)
            if "sexe" in tabular_features:
                n_tabular_features += 1 # Sex is one-hot encoded
            if pooling == "attention":
                self.model = AttentionNetWithMetadata(
                    n_tabular_variables=n_tabular_features,
                    **model_spec,
                )
            elif pooling == "avg":
                print("Using non-attention pooling!")
                self.model = CytoNetWithMetadata(
                    n_tabular_variables=n_tabular_features,
                    **model_spec,
                )
            else:
                raise NotImplementedError()
    
    def __call__(self, X, tabular_features=None):
        if isinstance(self.model, AttentionNetWithMetadata):
            return self.model(X, tabular_features)
        else:
            return self.model(X)

    def eval(self):
        self.model.eval()
