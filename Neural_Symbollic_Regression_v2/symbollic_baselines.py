"""
symbolic_baselines.py

Baseline symbolic regression models.

Author: Ravi Kumar U
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pysindy as ps
from pysr import PySRRegressor
from gplearn.genetic import SymbolicRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs import PySRConfig, GPLearnConfig

#result container
@dataclass
class BaselineResult:
    name: str
    equation: str
    rmse: float
    mae: float
    r2: float
    train_time: float
    success: bool


#abstract base class
class SymbolicBaseline(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.train_time = 0.0

    # ------------------------------------------------------
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None):
        pass

    # ------------------------------------------------------
    @abstractmethod
    def predict(self, X):
        pass

    # ------------------------------------------------------
    @abstractmethod
    def get_equation(self):
        pass

    # ------------------------------------------------------
    @abstractmethod
    def get_active_terms(self):
        pass

    # ------------------------------------------------------
    @abstractmethod
    def get_library_size(self):
        pass

    # ------------------------------------------------------
    @abstractmethod
    def get_model_name(self):
        pass

    # ------------------------------------------------------
    @abstractmethod
    def get_expression_complexity(self):
        pass

    # ------------------------------------------------------
    def evaluate(self, X, y):
        y_pred = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return rmse, mae, r2

    # ------------------------------------------------------
    def __repr__(self):
        return self.name


#SINDy baseline
class SINDyBaseline(SymbolicBaseline):
    def __init__(
            self,
            optimizer: Optional[ps.optimizers.BaseOptimizer] = None,
            feature_library: Optional[ps.feature_library.BaseFeatureLibrary] = None,
        ):
        super().__init__("SINDy")

        self.optimizer = optimizer or ps.STLSQ(threshold=0.1, alpha=0.05)
        self.feature_library = feature_library or ps.PolynomialLibrary(degree=3, include_bias=False)
        self.model = ps.SINDy(optimizer=self.optimizer, feature_library=self.feature_library)

    def fit(self, X, y, X_val=None, y_val=None):
        start = time.perf_counter()

        #we treat y as the system state.
        #dummy time vector is used avoid to numerical differentiation.
        t = np.arange(len(y))

        self.model.fit(x=y.reshape(-1, 1), t=t)
        self.train_time = (time.perf_counter() - start)

        return self

    # ------------------------------------------------------

    def predict(self, X):
        #SINDy predicts state derivatives or future trajectories, not arbitrary regression outputs.
        # Placeholder until regression variant is implemented.

        raise NotImplementedError(
            "Prediction for static regression benchmarks "
            "will be implemented in Part 2."
        )

    # ------------------------------------------------------
    def get_model_name(self):
        return self.name

    # ------------------------------------------------------
    def get_expression_complexity(self):
        return self.get_active_terms()
    
    # ------------------------------------------------------
    def get_equation(self):
        eq = self.model.equations()
        if len(eq):
            return eq[0]
        return ""

    # ------------------------------------------------------
    def get_active_terms(self):
        coef = self.model.coefficients()
        return int(np.count_nonzero(np.abs(coef) > 1e-10))

    # ------------------------------------------------------
    def get_library_size(self):
        coef = self.model.coefficients()
        return coef.shape[1]
    


# PySR Baseline
class PySRBaseline(SymbolicBaseline):
    def __init__(self, config: PySRConfig = PySRConfig()):
        super().__init__("PySR")
        self.cfg = config
        self.model = PySRRegressor(

            #search
            niterations=self.cfg.niterations,
            populations=self.cfg.populations,
            population_size=self.cfg.population_size,
            maxsize=self.cfg.maxsize,
            maxdepth=self.cfg.maxdepth,

            #operators
            binary_operators=self.cfg.binary_operators,
            unary_operators=self.cfg.unary_operators,

            #loss
            elementwise_loss=self.cfg.elementwise_loss,

            #parallel
            parallelism = self.cfg.parallelism,
            verbosity=self.cfg.verbosity,

            #reproducibility
            random_state=self.cfg.random_state,
            deterministic=self.cfg.deterministic
        )

    # ------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        start = time.perf_counter()
        self.model.fit(X, y)
        self.train_time = (time.perf_counter() - start)

        return self

    # ------------------------------------------------------
    def predict(self, X):
        return self.model.predict(X)

    # ------------------------------------------------------
    def get_equation(self):
        try:
            eq = self.model.get_best()["sympy_format"]
            return str(eq)

        except Exception:
            return ""

    # ------------------------------------------------------
    def get_active_terms(self):
        eq = self.get_equation()
        operators = [
            "+",
            "-",
            "*",
            "/",
            "sin",
            "cos",
            "exp",
            "log",
            "sqrt",
        ]
        count = 0
        for op in operators:
            count += eq.count(op)

        return count

    # ------------------------------------------------------
    def get_library_size(self):
        try:
            return int(self.model.get_best()["complexity"])

        except Exception:
            return -1

    # ------------------------------------------------------
    def get_model_name(self):
        return self.name
    
    # ------------------------------------------------------
    def get_expression_complexity(self):
        try:
            return int(self.model.get_best()["complexity"])
        except Exception:
            return -1
    


#GPLearn baseline
class GPLearnBaseline(SymbolicBaseline):
    def __init__(self, config: GPLearnConfig | None = None):
        super().__init__("GPLearn")
        self.cfg = config or GPLearnConfig()
        self.model = SymbolicRegressor(
            # Population
            population_size=self.cfg.population_size,
            generations=self.cfg.generations,
            tournament_size=self.cfg.tournament_size,
            stopping_criteria=self.cfg.stopping_criteria,

            # Functions
            function_set=self.cfg.function_set,

            # Tree Constraints
            init_depth=self.cfg.init_depth,
            init_method=self.cfg.init_method,
            const_range=self.cfg.const_range,
            parsimony_coefficient=self.cfg.parsimony_coefficient,

            # Reproducibility
            random_state=self.cfg.random_state,
            verbose=self.cfg.verbose,
        )

    # ------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        start = time.perf_counter()
        self.model.fit(X, y)
        self.train_time = (time.perf_counter() - start)

        return self

    # ------------------------------------------------------
    def predict(self, X):
        return self.model.predict(X)

    # ------------------------------------------------------
    def get_equation(self):
        try:
            return str(self.model._program)

        except Exception:
            return ""

    # ------------------------------------------------------
    def get_active_terms(self):
        eq = self.get_equation()
        operators = [
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "log",
            "abs",
            "neg",
            "inv",
            "sin",
            "cos",
            "tan",
        ]
        count = 0

        for op in operators:
            count += eq.count(op)

        return count

    # ------------------------------------------------------
    def get_library_size(self):
        try:
            return len(self.model._program.program)

        except Exception:
            return -1

    # ------------------------------------------------------
    def get_model_name(self):
        return self.name
    
    # ------------------------------------------------------
    def get_expression_complexity(self):
        try:
            return self.model._program.length_
        except Exception:
            return -1
    

#registry
BASELINES = {
    "sindy": SINDyBaseline,
    "pysr": PySRBaseline,
    "gplearn": GPLearnBaseline,
}


def list_available_baselines():
    return list(BASELINES.keys())


def create_baseline(name, **kwargs):
    name = name.lower()
    if name not in BASELINES:
        raise ValueError(
            f"Unknown baseline '{name}'. "
            f"Available: {list(BASELINES.keys())}"
        )

    return BASELINES[name](**kwargs)