from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict, field
from itertools import combinations
from typing import Callable, List, Sequence, Tuple, Dict, Optional
from copy import deepcopy

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

from configs import LibraryConfig, ModelConfig, TrainingConfig


try:
    from ray import tune
    _HAS_RAY = True
except Exception:
    _HAS_RAY = False


def set_seed(seed: int = 1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def id_fn(x):
    return x

def cube_fn(x):
    return x ** 3

def safe_log1p(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.clip(arr, -0.999999, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log1p(arr)
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    return out

def safe_sqrt(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.clip(x, 0.0, None))


def safe_inverse(x):
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-6
    return 1.0 / np.where(np.abs(x) < eps, np.sign(x) * eps + (x == 0) * eps, x)


def safe_inverse_square(x):
    inv = safe_inverse(x)
    return inv ** 2


def safe_abs(x):
    return np.abs(x)


def safe_sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


SAFE_FUNC_REGISTRY: Dict[str, Callable] = {
    "id": id_fn,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "square": np.square,
    "cube": cube_fn,
    "sqrt": safe_sqrt,
    "inverse": safe_inverse,
    "inverse_square": safe_inverse_square,
    "abs": safe_abs,
    "tanh": np.tanh,
    "sigmoid": safe_sigmoid,
    "log1p": safe_log1p
}

SYMPY_FUNC_MAP: Dict[str, Callable] = {
    "id": (lambda x: x),
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "sqrt": sp.sqrt,
    "inverse": lambda x: 1/x,
    "inverse_square": lambda x: 1/(x**2),
    "abs": sp.Abs,
    "tanh": sp.tanh,
    "sigmoid": lambda x: 1/(1+sp.exp(-x)),
    "log1p": lambda x: sp.log(x+1),
}



class FeatureLibrary:
    def __init__(self, config: LibraryConfig):
        self.cfg = config
        self._last_shape: Optional[Tuple[int, int]] = None
        self._last_names: Optional[List[str]] = None

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if X.ndim == 1:
            X = X[:, None]
        n_samples, n_features = X.shape

        funcs = self.cfg.funcs

        func_names = [fn if isinstance(fn, str) else getattr(fn, "__name__", str(fn)) for fn in funcs]
        callables = [SAFE_FUNC_REGISTRY.get(fn, fn) for fn in funcs]

        terms = []
        names = []

        def single(f, fname, i):
            return SAFE_NUMERIC(f(X[:, i])), f"{fname}(x{i})"


        nj = self.cfg.n_jobs if (self.cfg.n_jobs and self.cfg.n_jobs != 0) else 1
        single_results = Parallel(n_jobs=nj, backend="threading")(
            delayed(single)(f, fname, i)
            for fname, f in zip(func_names, callables)
            for i in range(n_features)
        )
        for t, name in single_results:
            terms.append(t)
            names.append(name)


        if self.cfg.include_interactions and n_features > 1 and self.cfg.max_order >= 2:
            for order in range(2, self.cfg.max_order + 1):
                feat_combos = list(combinations(range(n_features), order))
                func_combos = list(combinations(range(len(callables)), order))

                def interact(combo, fidxs):
                    term = np.ones(n_samples)
                    name_parts = []
                    for idx, fidx in zip(combo, fidxs):
                        f = callables[fidx]
                        fname = func_names[fidx]
                        term = term * SAFE_NUMERIC(f(X[:, idx]))
                        name_parts.append(f"{fname}(x{idx})")
                    return term, "*".join(name_parts)

                inter_results = Parallel(n_jobs=nj, backend="threading")(
                    delayed(interact)(c, fc) for c in feat_combos for fc in func_combos
                )
                for t, name in inter_results:
                    terms.append(t)
                    names.append(name)

        Phi = (
            np.stack(terms, axis=1).astype(np.float32)
            if terms
            else np.zeros((n_samples, 0), dtype=np.float32)
        )

        Phi = np.nan_to_num(
            Phi,
            copy=False,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        if Phi.shape[1] > 0:
            scaler = StandardScaler()
            Phi = scaler.fit_transform(Phi).astype(np.float32)
            self._scaler = scaler

        self._last_shape = Phi.shape
        self._last_names = names
        return Phi, names

    @property
    def last_feature_names(self) -> List[str]:
        if self._last_names is None:
            raise RuntimeError("Library has not been transformed yet.")
        return self._last_names


def SAFE_NUMERIC(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return arr


class MLP(nn.Module):
    def __init__(self, in_dim: int, cfg: ModelConfig):
        super().__init__()
        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }[cfg.activation]

        layers: List[nn.Module] = []
        hidden = cfg.hidden
        depth = max(int(cfg.depth), 1)

        layers += [nn.Linear(in_dim, hidden), act_layer()]
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))

        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act_layer()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))

        layers += [nn.Linear(hidden, cfg.out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)



class NeuralSRModel:
    def __init__(
            self,
            library_cfg: LibraryConfig,
            model_cfg: ModelConfig,
            train_cfg: TrainingConfig,
            extractor_alpha: float = 1e-3,
            extractor_method="lasso",
        ):

        self.library_cfg = library_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.extractor_alpha = extractor_alpha
        self.extractor_method = extractor_method

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        self.model = None

        self.X_train = None
        self.y_train = None
        self.Phi_train = None

        self.library = None
        self.trainer = None
        self.extractor = None

        self.feature_names = None
        self.coefficients = None
        self.symbolic_equation = None
        self.active_terms = None

    def build(self, input_dim: int):
        self.model = MLP(input_dim, self.model_cfg).to(self.device)

    def predict_from_phi(self, Phi: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.tensor(Phi, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.model(x).detach().cpu().numpy()
        return y

    def predict(self, X):
        Phi, _ = self.library.transform(X)
        return self.predict_from_phi(Phi)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.library = FeatureLibrary(self.library_cfg)
        # library_cfg = deepcopy(self.library_cfg)
        # if not self.ablation_cfg.use_nonlinear_library:
        #     library_cfg.funcs = ["id"]
        #     library_cfg.include_interactions = False
        # self.library = FeatureLibrary(library_cfg)

        Phi_train, self.feature_names = self.library.transform(X_train)

        #storing training data for post-training statistical analysis
        self.X_train = X_train
        self.y_train = y_train
        self.Phi_train = Phi_train

        self.build(Phi_train.shape[1])

        self.trainer = Trainer(self, self.train_cfg)
        if X_val is not None and y_val is not None:
            Phi_val, _ = self.library.transform(X_val)
            self.trainer.fit(Phi_train, y_train, Phi_val, y_val, verbose=False)

        else:
            self.trainer.fit(Phi_train, y_train, verbose=False,)

        # self.extractor = EquationExtractor(alpha=self.extractor_alpha, method="lasso")
        # method = self.ablation_cfg.sparse_method
        self.extractor = EquationExtractor(alpha=self.extractor_alpha, method=self.extractor_method)

        _, self.coefficients = self.extractor.fit_predict(Phi_train, y_train)

        self.symbolic_equation = self.extractor.to_sympy(
            self.coefficients,
            self.feature_names,
            X_train.shape[1],
        )

        self.active_terms = self.extractor.get_active_term_names(self.coefficients, self.feature_names)

        return self
    
    def get_feature_library(self):
        return self.Phi_train, self.feature_names
    
    def get_training_data(self):
        return self.X_train, self.y_train
    
    def get_equation(self):
        return str(self.symbolic_equation)
    
    def get_active_terms(self):
        return len(self.active_terms)
    
    def get_library_size(self):
        return len(self.feature_names)
    
    def get_expression_complexity(self):
        return len(self.active_terms)
    
    def get_model_name(self):
        return "NeuralSR"
    
    def get_statistics(self):
        from statistical_significance import StatisticalSignificance
        stats = StatisticalSignificance(self)
        stats.fit()
        return stats



class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.count = 0
        self.best_state = None

    def step(self, value: float, model: nn.Module) -> bool:
        if value + self.min_delta < self.best:
            self.best = value
            self.count = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.count += 1
            return self.count > self.patience



class Trainer:
    def __init__(self, model: NeuralSRModel, train_cfg: TrainingConfig):
        self.model = model
        self.cfg = train_cfg
        self.history: Dict[str, List[float]] = {"train": [], "val": []}


        use_amp = self._use_amp()
        if self.model.device.type == "cuda":

            self.scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
        else:

            class DummyScaler:
                def scale(self, loss): return loss
                def step(self, opt): opt.step()
                def update(self): pass
                def unscale_(self, opt): pass
            self.scaler = DummyScaler()

    def _use_amp(self) -> bool:
        if self.cfg.use_amp is not None:
            return bool(self.cfg.use_amp) and (self.model.device.type == "cuda")
        return self.model.device.type == "cuda"

    def _make_loader(self, Phi: np.ndarray, y: np.ndarray, shuffle=True):
        x = torch.tensor(Phi, dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(x, t)
        return torch.utils.data.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

    def fit(self,
            Phi_train: np.ndarray, y_train: np.ndarray,
            Phi_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: bool = True):
        assert self.model.model is not None, "Call model.build(input_dim) first."

        device = self.model.device
        net = self.model.model
        wd = self.model.model_cfg.weight_decay
        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=wd)
        loss_fn = nn.MSELoss()

        sched = None

        es = EarlyStopping(self.cfg.early_stop_patience, self.cfg.early_stop_delta)

        loader_tr = self._make_loader(Phi_train, y_train, shuffle=True)
        if Phi_val is not None and y_val is not None:
            loader_val = self._make_loader(Phi_val, y_val, shuffle=False)
        else:
            loader_val = None

        if self.cfg.scheduler == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.cfg.lr,
                steps_per_epoch=len(loader_tr), epochs=self.cfg.epochs,
                anneal_strategy='cos', pct_start=0.15
            )
        elif self.cfg.scheduler == "steplr":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.cfg.step_size, gamma=self.cfg.gamma)
        else:
            sched = None

        use_amp = self._use_amp()
        net.to(device)

        for epoch in range(self.cfg.epochs):
            net.train()
            epoch_losses = []
            for xb, yb in loader_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                if use_amp and isinstance(self.scaler, torch.amp.GradScaler):
                    with torch.cuda.amp.autocast():
                        pred = net(xb)
                        loss = loss_fn(pred, yb)
                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clip is not None:
                        self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    pred = net(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    if self.cfg.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
                    opt.step()
                epoch_losses.append(loss.detach().item())
            train_loss = float(np.mean(epoch_losses))

            if loader_val is not None:
                net.eval()
                with torch.no_grad():
                    vlosses = []
                    for xb, yb in loader_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        pred = net(xb)
                        vlosses.append(loss_fn(pred, yb).item())
                val_loss = float(np.mean(vlosses))
                self.history["val"].append(val_loss)
                stop = es.step(val_loss, net)
            else:
                self.history["val"].append(float("nan"))
                stop = es.step(train_loss, net)

            self.history["train"].append(train_loss)

            if sched is not None:
                if self.cfg.scheduler == "onecycle":
                    sched.step()
                else:
                    sched.step()

            if verbose and (epoch % 10 == 0 or epoch == self.cfg.epochs - 1):
                msg = f"Epoch {epoch:04d} | train MSE {train_loss:.6f}"
                if loader_val is not None:
                    msg += f" | val MSE {val_loss:.6f}"
                print(msg)

            if stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best={es.best:.6f}).")
                break

        if es.best_state is not None:
            net.load_state_dict(es.best_state)

        return self.history


class EquationExtractor:
    def __init__(self, alpha: float = 1e-3, method: str = "lasso"):
        self.alpha = alpha
        self.method = method

    def fit_predict(self, Phi: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.method == "lasso":
            model = Lasso(alpha=self.alpha, max_iter=100000)
        elif self.method == "elastic":
            model = ElasticNet(alpha=self.alpha, l1_ratio=0.7, max_iter=100000)
        else:
            raise ValueError("Unknown method for symbolic regression coefficients")
        model.fit(Phi, y)
        return model.predict(Phi), model.coef_

    def to_sympy(self, coefs: np.ndarray, feature_names: List[str], n_vars: int) -> sp.Expr:
        xs = sp.symbols(f"x0:{n_vars}")
        expr = 0
        for c, name in zip(coefs, feature_names):
            if abs(float(c)) < 1e-8:
                continue
            term = self._name_to_sympy(name, xs)
            expr = expr + float(c) * term
        return sp.simplify(expr)

    def _name_to_sympy(self, name: str, xs: Sequence[sp.Symbol]) -> sp.Expr:
        parts = name.split("*")
        term = 1
        for p in parts:
            fname, arg = p.split("(")
            idx = int(arg.replace("x", "").replace(")", ""))
            fn = SYMPY_FUNC_MAP[fname]
            term = term * fn(xs[idx])
        return term

    def get_active_term_names(self, coefs: np.ndarray, feature_names: List[str], threshold: float = 1e-6) -> List[str]:
        active = [name for c, name in zip(coefs, feature_names) if abs(float(c)) > threshold]
        return active



def compute_ground_truth_coeffs_via_library(true_expr: sp.Expr,
                                            variables: List[sp.Symbol],
                                            library: FeatureLibrary,
                                            sample_n: int = 2000,
                                            sample_low: float = -1.0,
                                            sample_high: float = 1.0,
                                            lasso_alpha: float = 1e-6,
                                            seed: Optional[int] = 1337) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    n_vars = len(variables)
    X = rng.uniform(sample_low, sample_high, size=(sample_n, n_vars))
    f = sp.lambdify(variables, true_expr, modules=["numpy"])
    y = f(*[X[:, i] for i in range(n_vars)])
    y = np.asarray(y).squeeze()

    Phi, names = library.transform(X)
    if Phi.size == 0:
        return np.array([]), names

    try:
        lasso = Lasso(alpha=lasso_alpha, max_iter=10000)
        lasso.fit(Phi, y)
        coef_true = lasso.coef_
    except Exception:
        coef_true, *_ = np.linalg.lstsq(Phi, y, rcond=None)

    return np.asarray(coef_true), names


def compare_supports(coef_true: np.ndarray, coef_extracted: np.ndarray, feature_names: List[str],
                     threshold: float = 1e-6) -> Dict[str, object]:
    assert coef_true.shape == coef_extracted.shape, "Coefficient shapes must match"
    true_support = set(i for i, c in enumerate(coef_true) if abs(float(c)) > threshold)
    ext_support = set(i for i, c in enumerate(coef_extracted) if abs(float(c)) > threshold)

    tp = sorted(list(true_support & ext_support))
    fn = sorted(list(true_support - ext_support))
    fp = sorted(list(ext_support - true_support))

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    result = {
        "tp_idx": tp,
        "fp_idx": fp,
        "fn_idx": fn,
        "tp_names": [feature_names[i] for i in tp],
        "fp_names": [feature_names[i] for i in fp],
        "fn_names": [feature_names[i] for i in fn],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_true_nonzero": len(true_support),
        "n_extracted_nonzero": len(ext_support),
    }
    return result



class Tuner:
    def __init__(self, lib: FeatureLibrary, base_model_cfg: ModelConfig, base_train_cfg: TrainingConfig):
        self.lib = lib
        self.mc = base_model_cfg
        self.tc = base_train_cfg
        if not _HAS_RAY:
            raise ImportError("Ray Tune is not available. Install ray[tune] to use Tuner.")

    def _objective(self, config, X_train, y_train, X_val, y_val):
        Phi_tr, _ = self.lib.transform(X_train)
        Phi_va, _ = self.lib.transform(X_val)

        mcfg = ModelConfig(
            hidden=config["hidden"],
            depth=config["depth"],
            dropout=config["dropout"],
            activation=config["activation"],
            out_dim=1,
            weight_decay=config["weight_decay"],
        )
        model = NeuralSRModel(mcfg)
        model.build(Phi_tr.shape[1])

        tcfg = TrainingConfig(
            lr=config["lr"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            grad_clip=1.0,
            early_stop_patience=25,
            scheduler="onecycle",
        )
        trainer = Trainer(model, tcfg)
        trainer.fit(Phi_tr, y_train, Phi_va, y_val, verbose=False)

        pred = model.predict_from_phi(Phi_va)
        rmse_val = float(np.sqrt(mean_squared_error(y_val, pred)))
        tune.report({"rmse": rmse_val})

    def run(self, X, y, val_split: float = 0.2, num_samples: int = 20):
        assert _HAS_RAY, "Ray Tune not available"
        n = len(X)
        idx = np.arange(n)
        np.random.shuffle(idx)
        cut = int(n * (1 - val_split))
        tr, va = idx[:cut], idx[cut:]
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        search_space = {
            "hidden": tune.choice([64, 128, 256]),
            "depth": tune.choice([1, 2, 3, 4]),
            "dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
            "activation": tune.choice(["relu", "gelu", "tanh"]),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "lr": tune.loguniform(5e-5, 5e-3),
            "batch_size": tune.choice([64, 128, 256]),
            "epochs": tune.choice([100, 150, 200]),
        }
    
        tuner = tune.Tuner(
            tune.with_parameters(self._objective, X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va),
            param_space=search_space,
            tune_config=tune.TuneConfig(num_samples=num_samples, metric="rmse", mode="min"),
            run_config=tune.RunConfig(name="NSR_OOP_Tuning")
        )
        results = tuner.fit()
        return results



def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def train_val_split(X, y, val_split=0.2, seed=1337):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - val_split))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]

