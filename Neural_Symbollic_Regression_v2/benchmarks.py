"""
benchmarks.py

Benchmark definitions for Neural Symbolic Regression.

Currently supported benchmark suites:
    - Nguyen
    - Keijzer
    - Vladislavleva
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import sympy as sp


class Benchmarks:

    # nguyen benchmark setup
    @staticmethod
    def nguyen(n: int = 1):
        x, y, z, w = sp.symbols("x y z w")
        benchmarks = {
            1: (x**3 + x**2 + x, [x]),
            2: (x**4 + x**3 + x**2 + x, [x]),
            3: (x**5 + x**4 + x**3 + x**2 + x, [x]),
            4: (sp.sin(x) + sp.sin(x + x**2), [x]),
            5: (sp.exp(x) - sp.exp(-x), [x]),
            6: (sp.log(x + 1) + sp.log(x**2 + 1), [x]),
            7: (sp.sin(x) + sp.sin(y**2), [x, y]),
            8: (sp.exp(x + y), [x, y]),
            9: (x * y + sp.sin(x) * sp.cos(y), [x, y]),
            10: (x * y + z**2, [x, y, z]),
            11: (sp.sin(x) + sp.cos(y) + z, [x, y, z]),
            12: (x**2 + y**2 + z**2, [x, y, z]),
            13: (x + y + z + w, [x, y, z, w]),
            14: (x * y + z * w, [x, y, z, w]),
            15: (sp.sin(x) + sp.cos(y) + z**2 - sp.sqrt(w), [x, y, z, w]),
        }

        return benchmarks[n]

    # keijzer benchmark setup
    @staticmethod
    def keijzer(n: int):
        x, y, z = sp.symbols("x y z")
        benchmarks = {
            # keijzer-1
            1: (
                0.3 * x * sp.sin(2 * sp.pi * x),
                [x],
            ),
            # keijzer-2
            2: (
                x**2
                * sp.exp(-x)
                * sp.cos(x)
                * sp.sin(x)
                * (sp.sin(x) ** 2 * sp.cos(x) - 1),
                [x],
            ),
            # # keijzer-5
            # 5: (
            #     (30 * x * z) / ((x - 10) * y**2),
            #     [x, y, z],
            # ),
        }

        return benchmarks[n]

    # vladislavleva benchmark setup
    @staticmethod
    def vladislavleva(n: int):
        x, y = sp.symbols("x y")
        benchmarks = {

            # vladislavleva-1
            1: (
                sp.exp(-(x - 1) ** 2) / (1.2 + (y - 2.5) ** 2),
                [x, y],
            ),

            # vladislavleva-4
            4: (
                10 / (5 + (x - 3) ** 2 + (y - 3) ** 2),
                [x, y],
            ),
        }

        return benchmarks[n]

    # benchmark Loader
    @staticmethod
    def get(family: str, number: int):
        family = family.lower()
        mapping = {
            "nguyen": Benchmarks.nguyen,
            "keijzer": Benchmarks.keijzer,
            "vladislavleva": Benchmarks.vladislavleva,
        }

        if family not in mapping:
            raise ValueError(f"Unknown benchmark family: {family}")

        return mapping[family](number)

    # benchmarks considered
    @staticmethod
    def list_available() -> Dict[str, List[int]]:
        return {
            "Nguyen": [1, 2, 3, 4],
            "Keijzer": [1, 2],
            "Vladislavleva": [1, 4],
        }

    # sampling domain
    @staticmethod
    def default_domain(family: str):
        family = family.lower()
        domains = {
            "nguyen": (-1.0, 1.0),
            "keijzer": (-1.0, 1.0),
            # shift because equations are centered around x≈3
            "vladislavleva": (0.0, 6.0),
        }

        return domains[family]

    # generating data
    @staticmethod
    def sample(expr, variables, n_samples=1000, noise=0.0, low=None, high=None, seed=None):
        rng = np.random.default_rng(seed)
        if low is None or high is None:
            low = -1.0
            high = 1.0
        n_vars = len(variables)
        X = rng.uniform(low, high,size=(n_samples, n_vars))
        f = sp.lambdify(variables, expr, modules=["numpy"])
        y = f(*[X[:, i] for i in range(n_vars)])
        y = np.asarray(y).squeeze()
        if noise > 0:
            y += rng.normal(0, noise, size=y.shape)

        return X.astype(np.float32), y.astype(np.float32)


#quick test
if __name__ == "__main__":
    print("Available Benchmarks\n")

    for family, ids in Benchmarks.list_available().items():
        print(f"{family}: {ids}")

    print("\nTesting benchmark loading...\n")
    expr, vars_ = Benchmarks.get("Keijzer", 2)
    print("Expression:")
    print(expr)

    X, y = Benchmarks.sample(expr, vars_, n_samples=10, noise=0.01, low=-1, high=1, seed=42)
    print("\nSample Shapes")
    print("X:", X.shape)
    print("y:", y.shape)