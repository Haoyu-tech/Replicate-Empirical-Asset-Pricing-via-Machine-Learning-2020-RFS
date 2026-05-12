"""
Model definitions for Gu, Kelly, Xiu (2020) replication.
All models implement fit(X_train, y_train) and predict(X_test).
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, SGDRegressor
)
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Utility ─────────────────────────────────────────────────────────────────

def r2_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R² = 1 - SS_res / SS_tot (benchmark = 0)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.35) -> float:
    r = y_true - y_pred
    loss = np.where(
        np.abs(r) <= delta,
        0.5 * r ** 2,
        delta * (np.abs(r) - 0.5 * delta)
    )
    return float(np.sum(loss))


# ── Linear models ────────────────────────────────────────────────────────────

class OLS3Model:
    """OLS with only 3 predictors: size (me), value (be_me), momentum (ret_12_1)."""
    THREE_COLS = ['me', 'be_me', 'ret_12_1']

    def __init__(self):
        self.model = LinearRegression()
        self.col_idx = None

    def fit(self, X_train, y_train, col_names=None):
        if col_names is not None:
            self.col_idx = [list(col_names).index(c) for c in self.THREE_COLS
                            if c in col_names]
        X3 = X_train[:, self.col_idx] if self.col_idx else X_train[:, :3]
        self.model.fit(X3, y_train)

    def predict(self, X_test):
        X3 = X_test[:, self.col_idx] if self.col_idx else X_test[:, :3]
        return self.model.predict(X3)


class OLSModel:
    """OLS with all features (no regularisation)."""
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y, **kw):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class OLSHuberModel:
    """OLS with Huber loss (robust regression)."""
    def __init__(self, epsilon=1.35, alpha=1e-4):
        self.epsilon = epsilon
        self.alpha = alpha
        self.model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)

    def fit(self, X, y, **kw):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ENetModel:
    """Elastic net with L1/L2 penalty. Huber variant uses SGD."""
    def __init__(self, alpha=0.001, l1_ratio=0.5, huber=False):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.huber = huber
        if huber:
            self.model = SGDRegressor(
                loss='huber', epsilon=1.35,
                penalty='elasticnet', l1_ratio=l1_ratio,
                alpha=alpha, max_iter=2000, tol=1e-4,
                learning_rate='optimal'
            )
        else:
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                    max_iter=5000, tol=1e-4)

    def fit(self, X, y, **kw):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class PCRModel:
    """Principal Component Regression."""
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.reg = LinearRegression()

    def fit(self, X, y, **kw):
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)
        self.reg.fit(Xp, y)

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)
        return self.reg.predict(Xp)


class PLSModel:
    """Partial Least Squares Regression."""
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.model = PLSRegression(n_components=n_components, max_iter=1000)

    def fit(self, X, y, **kw):
        self.model.fit(X, y.reshape(-1, 1))

    def predict(self, X):
        return self.model.predict(X).ravel()


# ── Tree models ───────────────────────────────────────────────────────────────

class GBRTModel:
    """Gradient Boosted Regression Trees. huber=True uses Huber loss."""
    def __init__(self, n_estimators=300, max_depth=2, learning_rate=0.01,
                 min_samples_leaf=1000, max_features='sqrt', huber=True):
        loss = 'huber' if huber else 'squared_error'
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            loss=loss,
            subsample=0.5,
            random_state=42,
        )

    def fit(self, X, y, **kw):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_


class RFModel:
    """Random Forest Regressor."""
    def __init__(self, n_estimators=300, max_depth=None,
                 min_samples_leaf=1000, max_features='sqrt'):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X, y, **kw):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_


# ── Neural Networks ───────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _NNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.linear(self.bn(x))))


class _NNArchitecture(nn.Module):
    """Feedforward NN with BN→Linear→ReLU→Dropout blocks.
    Architecture: [P, 32, 16, 8, 4, 2] truncated at n_layers.
    """
    WIDTHS = [32, 16, 8, 4, 2]

    def __init__(self, in_dim, n_layers, dropout=0.5):
        super().__init__()
        widths = self.WIDTHS[:n_layers]
        dims = [in_dim] + widths
        blocks = []
        for i in range(len(widths)):
            blocks.append(_NNBlock(dims[i], dims[i+1], dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out_bn = nn.BatchNorm1d(widths[-1])
        self.out = nn.Linear(widths[-1], 1)

    def forward(self, x):
        h = self.blocks(x)
        return self.out(self.out_bn(h)).squeeze(-1)


class NNModel:
    """Neural network (n_layers = 1..5) matching Gu et al. architecture."""

    def __init__(self, n_layers=3, lr=1e-3, l1_lambda=1e-5,
                 epochs=100, batch_size=10000, patience=5):
        self.n_layers = n_layers
        self.lr = lr
        self.l1_lambda = l1_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.net = None
        self.scaler = StandardScaler()

    def _to_tensor(self, X, y=None):
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
            return Xt, yt
        return Xt

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kw):
        # Standardize inputs (characteristics are already in [-1,1],
        # but scaling helps convergence)
        X_train = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)

        in_dim = X_train.shape[1]
        self.net = _NNArchitecture(in_dim, self.n_layers).to(DEVICE)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=0)
        loss_fn = nn.MSELoss()

        Xt, yt = self._to_tensor(X_train, y_train)
        dataset = TensorDataset(Xt, yt)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val = np.inf
        wait = 0
        best_state = None

        for epoch in range(self.epochs):
            self.net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                # L1 regularization on first layer
                l1 = sum(p.abs().sum() for p in
                         list(self.net.blocks[0].parameters()))
                (loss + self.l1_lambda * l1).backward()
                optimizer.step()

            # Validation
            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    Xv, yv = self._to_tensor(X_val, y_val)
                    val_loss = loss_fn(self.net(Xv), yv).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in
                                  self.net.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, X):
        X = self.scaler.transform(X)
        self.net.eval()
        with torch.no_grad():
            Xt = self._to_tensor(X)
            return self.net(Xt).cpu().numpy()


# ── Hyperparameter grid search ───────────────────────────────────────────────

def tune_and_fit(ModelClass, param_grid: list, X_train, y_train, X_val, y_val,
                 col_names=None) -> object:
    """Grid-search on validation set (Huber loss), return best fitted model."""
    best_loss = np.inf
    best_model = None
    for params in param_grid:
        try:
            m = ModelClass(**params)
            if col_names is not None and hasattr(m, 'fit'):
                import inspect
                sig = inspect.signature(m.fit)
                if 'col_names' in sig.parameters:
                    m.fit(X_train, y_train, col_names=col_names)
                elif 'X_val' in sig.parameters:
                    m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                else:
                    m.fit(X_train, y_train)
            else:
                m.fit(X_train, y_train)
            val_pred = m.predict(X_val)
            loss = huber_loss(y_val, val_pred)
            if loss < best_loss:
                best_loss = loss
                best_model = m
        except Exception as e:
            continue

    # Refit on train + val combined
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    if best_model is not None:
        # Get params from best model
        params = best_model.__dict__.copy()
        # Refit same class with same params on full data
        try:
            m_final = ModelClass(**{k: v for k, v in params.items()
                                    if not k.startswith('model') and
                                    not k.startswith('scaler') and
                                    not k.startswith('pca') and
                                    not k.startswith('reg') and
                                    not k.startswith('net') and
                                    not k.startswith('col_idx')})
        except Exception:
            m_final = best_model
        try:
            if hasattr(m_final, 'fit'):
                m_final.fit(X_tv, y_tv)
        except Exception:
            m_final = best_model
        return m_final
    return best_model


# ── Param grids ─────────────────────────────────────────────────────────────

ENET_GRID = [
    {'alpha': a, 'l1_ratio': r, 'huber': True}
    for a in [1e-4, 1e-3, 1e-2, 1e-1]
    for r in [0.1, 0.5, 0.9]
]

PCR_GRID = [{'n_components': k} for k in [3, 5, 10, 20, 50]]

PLS_GRID = [{'n_components': k} for k in [1, 2, 3, 5]]

GBRT_GRID = [
    {'max_depth': d, 'learning_rate': lr, 'n_estimators': 300,
     'min_samples_leaf': leaf, 'max_features': 'sqrt', 'huber': True}
    for d in [1, 2, 3]
    for lr in [0.01, 0.1]
    for leaf in [500, 1000]
]

RF_GRID = [
    {'n_estimators': 300, 'max_depth': None,
     'min_samples_leaf': leaf, 'max_features': 'sqrt'}
    for leaf in [500, 1000, 2000]
]

NN_GRID = {
    n: [{'n_layers': n, 'lr': lr, 'l1_lambda': lam, 'epochs': 100,
         'batch_size': 10000, 'patience': 5}]
    for n in range(1, 6)
    for lr in [1e-3]
    for lam in [1e-5]
}
