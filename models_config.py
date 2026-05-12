"""
Model definitions for Gu, Kelly, Xiu (2020) replication.
Matches paper's exact model specifications.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import (
    LinearRegression, HuberRegressor, ElasticNet, Ridge, SGDRegressor
)
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Metrics ───────────────────────────────────────────────────────────────────

def r2_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def huber_loss_val(y_true, y_pred, delta=1.35):
    r = y_true - y_pred
    return float(np.where(np.abs(r) <= delta, 0.5*r**2,
                          delta*(np.abs(r) - 0.5*delta)).sum())


# ── Linear models ─────────────────────────────────────────────────────────────

class OLS3Model:
    """OLS with 3 predictors: mvel1 (size), bm (value), mom12m (momentum)."""
    THREE = ['mvel1', 'bm', 'mom12m']

    def __init__(self):
        self.model = LinearRegression()
        self._idx = None

    def fit(self, X, y, col_names=None):
        if col_names is not None:
            cn = list(col_names)
            self._idx = [cn.index(c) for c in self.THREE if c in cn]
        X3 = X[:, self._idx] if self._idx else X[:, :3]
        self.model.fit(X3, y); return self

    def predict(self, X):
        X3 = X[:, self._idx] if self._idx else X[:, :3]
        return self.model.predict(X3)


class OLSHuberModel:
    """
    OLS with Huber loss.
    Uses mini-batch SGD (no regularization) to avoid the ~8 GB float64 conversion
    that sklearn HuberRegressor requires when N × P is large.
    Data is already rank-normalised to [-1,1] so no scaling is needed.
    """
    def __init__(self, epsilon=1.35, alpha=0.0, max_iter=2000, batch_size=50000):
        self.epsilon   = epsilon
        self.alpha     = alpha
        self.max_iter  = max_iter
        self.batch_size = batch_size
        self.model = SGDRegressor(
            loss='huber', epsilon=epsilon,
            penalty='l2', alpha=alpha,        # alpha=0 → pure Huber loss (OLS+H)
            max_iter=max_iter, tol=1e-4,
            learning_rate='invscaling', eta0=0.01, power_t=0.25,
            random_state=42, shuffle=True,
        )

    def fit(self, X, y, **kw):
        X = X.astype('float32')
        # Mini-batch partial_fit to keep memory footprint small
        N = len(X)
        for _ in range(5):                    # 5 passes over the data
            idx = np.random.default_rng(42).permutation(N)
            for start in range(0, N, self.batch_size):
                sl = idx[start:start + self.batch_size]
                self.model.partial_fit(X[sl], y[sl])
        return self

    def predict(self, X):
        return self.model.predict(X.astype('float32'))


class ENetHuberModel:
    """Elastic Net with Huber loss via SGD (ENet+H)."""
    def __init__(self, alpha=1e-3, l1_ratio=0.5):
        self.alpha = alpha; self.l1_ratio = l1_ratio
        self.model = SGDRegressor(
            loss='huber', epsilon=1.35,
            penalty='elasticnet', l1_ratio=l1_ratio,
            alpha=alpha, max_iter=5000, tol=1e-5,
            learning_rate='invscaling', eta0=0.01, power_t=0.25,
            random_state=42,
        )

    def fit(self, X, y, **kw):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)


class PCRModel:
    """Principal Component Regression."""
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.scaler = StandardScaler(copy=False)
        self.pca    = PCA(n_components=n_components, random_state=42)
        self.reg    = LinearRegression()

    def fit(self, X, y, **kw):
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)
        self.reg.fit(Xp, y); return self

    def predict(self, X):
        return self.reg.predict(self.pca.transform(self.scaler.transform(X)))


class PLSModel:
    """Partial Least Squares Regression."""
    def __init__(self, n_components=3):
        self.model = PLSRegression(n_components=n_components, max_iter=1000)

    def fit(self, X, y, **kw):
        self.model.fit(X, y.reshape(-1, 1)); return self

    def predict(self, X):
        return self.model.predict(X).ravel()


# ── Tree models ───────────────────────────────────────────────────────────────

class GBRTHuberModel:
    """Gradient Boosting Trees with Huber loss (GBRT+H)."""
    def __init__(self, n_estimators=300, max_depth=2, learning_rate=0.01,
                 min_samples_leaf=1000, subsample=0.5):
        self.model = GradientBoostingRegressor(
            loss='huber', n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, min_samples_leaf=min_samples_leaf,
            subsample=subsample, max_features='sqrt', random_state=42,
        )
        self._fi = None

    def fit(self, X, y, **kw):
        self.model.fit(X, y)
        self._fi = self.model.feature_importances_; return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self._fi


class RFModel:
    """Random Forest Regressor."""
    def __init__(self, n_estimators=300, max_depth=None, min_samples_leaf=1000):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features='sqrt',
            n_jobs=-1, random_state=42,
        )
        self._fi = None

    def fit(self, X, y, **kw):
        self.model.fit(X, y)
        self._fi = self.model.feature_importances_; return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self._fi


# ── Neural Networks ───────────────────────────────────────────────────────────

class _NNArch(nn.Module):
    """
    Paper architecture (Section 3.5):
      NN1: P→32→1,  NN2: P→32→16→1,  NN3: P→32→16→8→1,
      NN4: P→32→16→8→4→1,  NN5: P→32→16→8→4→2→1
    BatchNorm + ReLU at each hidden layer. L1 on weights.
    """
    WIDTHS = [32, 16, 8, 4, 2]

    def __init__(self, in_dim: int, n_layers: int):
        super().__init__()
        widths = self.WIDTHS[:n_layers]
        layers, d = [], in_dim
        for w in widths:
            layers += [nn.BatchNorm1d(d), nn.Linear(d, w), nn.ReLU()]
            d = w
        layers += [nn.BatchNorm1d(d), nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NNModel:
    """Neural network ensemble (10 nets) as in the paper."""
    N_ENSEMBLE = 10

    def __init__(self, n_layers=3, lr=1e-3, l1_lambda=1e-5,
                 max_epochs=100, batch_size=10000, patience=5):
        self.n_layers   = n_layers
        self.lr         = lr
        self.l1_lambda  = l1_lambda
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.nets       = []
        self.scaler     = StandardScaler(copy=False)

    def _t(self, X, y=None):
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if y is not None:
            return Xt, torch.tensor(y, dtype=torch.float32, device=DEVICE)
        return Xt

    def _train_one(self, Xtr, ytr, Xva, yva, seed):
        torch.manual_seed(seed)
        net = _NNArch(Xtr.shape[1], self.n_layers).to(DEVICE)
        opt = optim.Adam(net.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        Xt, yt = self._t(Xtr, ytr)
        ld = DataLoader(TensorDataset(Xt, yt),
                        batch_size=self.batch_size, shuffle=True)
        Xv, yv = (self._t(Xva, yva) if Xva is not None else (None, None))
        best_val, wait, best_sd, best_ep = np.inf, 0, None, 0

        for ep in range(self.max_epochs):
            net.train()
            for xb, yb in ld:
                opt.zero_grad()
                loss = mse(net(xb), yb)
                l1 = sum(m.weight.abs().sum() for m in net.net
                         if isinstance(m, nn.Linear))
                (loss + self.l1_lambda * l1).backward()
                opt.step()
            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    vl = mse(net(Xv), yv).item()
                if vl < best_val:
                    best_val = vl
                    best_sd  = {k: v.clone() for k, v in net.state_dict().items()}
                    best_ep  = ep + 1
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_sd:
            net.load_state_dict(best_sd)
        return net, best_ep

    def fit(self, Xtr, ytr, Xva=None, yva=None, **kw):
        Xtr = self.scaler.fit_transform(Xtr.astype('float32'))
        if Xva is not None:
            Xva = self.scaler.transform(Xva.astype('float32'))
        results = [self._train_one(Xtr, ytr, Xva, yva, s)
                   for s in range(self.N_ENSEMBLE)]
        self.nets = [r[0] for r in results]
        eps = [r[1] for r in results]
        self._best_epoch = int(np.median(eps)) if eps else self.max_epochs
        return self

    def predict(self, X):
        X  = self.scaler.transform(X.astype('float32'))
        Xt = self._t(X)
        ps = []
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                ps.append(net(Xt).cpu().numpy())
        return np.mean(ps, axis=0)


# ── Hyperparameter grids ──────────────────────────────────────────────────────

ENET_GRID = [
    {'alpha': a, 'l1_ratio': r}
    for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for r in [0.1, 0.5, 0.9]
]

PCR_GRID  = [{'n_components': k} for k in [3, 5, 10, 20, 50]]
PLS_GRID  = [{'n_components': k} for k in [1, 2, 3, 5]]

GBRT_GRID = [
    {'max_depth': d, 'learning_rate': lr,
     'min_samples_leaf': leaf, 'n_estimators': 300}
    for d in [1, 2, 3, 4, 5, 6]
    for lr in [0.01, 0.1]
    for leaf in [500, 1000]
]

RF_GRID   = [{'n_estimators': 300, 'min_samples_leaf': leaf}
             for leaf in [500, 1000, 2000]]

NN_L1_GRID = [1e-5, 1e-4, 1e-3]
