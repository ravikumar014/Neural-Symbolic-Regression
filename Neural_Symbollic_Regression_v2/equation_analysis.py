"""
This code provides utilities for evaluating symbolic regression models based on equation recovery rather than prediction accuracy.

Capabilities
------------
- Equation normalization
- Symbolic equivalence checking
- Numerical equivalence checking
- Recovery classification
- Recovery summary generation

Author : Ravi Kumar U
Project: Neural Symbolic Regression Benchmark Framework
===============================================================
"""

from __future__ import annotations

from typing import Dict, Optional, List
import warnings

import numpy as np
import pandas as pd
import re
import sympy as sp

from sympy import (
    simplify,
    sympify,
    expand,
    factor,
    symbols,
    lambdify,
)
from sympy.core.sympify import SympifyError

warnings.filterwarnings("ignore")


class EquationAnalyzer:
    def __init__(self, results_df: pd.DataFrame): #, true_equations: Dict[str, str]
        self.df = results_df.copy()
        # self.true_equations = true_equations
        self.x = symbols("x")
        self.recovery_df = None
        self.status_priority = {
            "Exact": 4,
            "Equivalent": 3,
            "Approximate": 2,
            "Failed": 1,
        }

    #internal utilities
    def normalize_equation(self, equation: str):
        if equation is None:
            return None

        if pd.isna(equation):
            return None


        # GPLearn variables
        equation = re.sub(r"\bX(\d+)\b", r"x\1", equation)

        # Single-variable benchmark compatibility
        equation = re.sub(r"\bx0\b", "x", equation)
        equation = str(equation).strip()

        if equation == "":
            return None

        #common operator conversions
        replacements = {
            "^": "**",
            "ln(": "log(",
            "PI": "pi",
            "Pi": "pi",
            "E": "E",
        }

        for old, new in replacements.items():
            equation = equation.replace(old, new)

        # try:
        #     print(type(equation))
        #     print(equation)
        #     expr = sympify(equation, evaluate=True)
        #     expr = expand(expr)
        #     expr = simplify(expr)
        #     return expr

        # except (SympifyError, TypeError, ValueError):
        #     return None

        try:
            print("=" * 80)
            print("Input:", equation)

            # expr = sympify(equation, evaluate=True)
            locals_dict = {
                "add": lambda a, b: a + b,
                "sub": lambda a, b: a - b,
                "mul": lambda a, b: a * b,
                "div": lambda a, b: a / b,
                "neg": lambda a: -a,
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "sqrt": sp.sqrt,
                "log": sp.log,
                "exp": sp.exp,
                "Abs": sp.Abs,
            }

            expr = sympify(
                equation,
                locals=locals_dict,
                evaluate=True,
            )

            print("Sympified type:", type(expr))
            print("Sympified value:", expr)

            expr = expand(expr)
            expr = simplify(expr)

            return expr

        except Exception as e:
            print("FAILED:", equation)
            print(type(e))
            print(e)
            return None
        

    def symbolic_equivalence(self, predicted: str, ground_truth: str) -> bool:
        pred_expr = self.normalize_equation(predicted)
        true_expr = self.normalize_equation(ground_truth)

        if pred_expr is None or true_expr is None:
            return False

        try:
            difference = simplify(pred_expr - true_expr)
            return difference == 0

        except Exception:
            return False
    

    def numerical_equivalence(self, predicted: str, ground_truth: str, n_samples: int = 1000, tolerance: float = 1e-6, domain: tuple = (-5.0, 5.0)) -> bool:
        pred_expr = self.normalize_equation(predicted)
        true_expr = self.normalize_equation(ground_truth)

        if pred_expr is None or true_expr is None:
            return False

        try:
            pred_func = lambdify(self.x, pred_expr, "numpy")
            true_func = lambdify(self.x, true_expr, "numpy")
            rng = np.random.default_rng(42)
            x = rng.uniform(domain[0], domain[1], n_samples)

            y_pred = pred_func(x)
            y_true = true_func(x)
            y_pred = np.asarray(y_pred, dtype=float)
            y_true = np.asarray(y_true, dtype=float)

            if (
                np.any(np.isnan(y_pred))
                or np.any(np.isnan(y_true))
                or np.any(np.isinf(y_pred))
                or np.any(np.isinf(y_true))
            ):
                return False

            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            return rmse <= tolerance

        except Exception:
            return False


    #classifying the recovery
    def classify_recovery(self, predicted: str, ground_truth: str) -> str:
        if predicted is None:
            return "Failed"

        if ground_truth is None:
            return "Failed"

        #symbolic equivalence
        if self.symbolic_equivalence(
            predicted,
            ground_truth,
        ):
            return "Exact"

        #numerical equivalence
        if self.numerical_equivalence(
            predicted,
            ground_truth,
        ):
            return "Approximate"

        #recovery failed
        return "Failed"


    #table generation
    def build_recovery_table(self) -> pd.DataFrame:
        if self.recovery_df is not None:
            return self.recovery_df
        
        records = []

        for _, row in self.df.iterrows():
            benchmark = row["benchmark"]
            model = row["model"]
            ground_truth = row["true_equation"]
            predicted = row["extracted_equation"]
            recovery = self.classify_recovery(predicted, ground_truth)

            records.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "family": row.get("family", None),
                    "rmse": row.get("rmse", np.nan),
                    "mae": row.get("mae", np.nan),
                    "r2": row.get("r2", np.nan),
                    "complexity": row.get("complexity", np.nan),
                    "true_equation": ground_truth,
                    "predicted_equation": predicted,
                    "recovery": recovery,
                }
            )
        recovery_df = pd.DataFrame(records)
        self.recovery_df = recovery_df
        return self.recovery_df
    

    def clear_cache(self):
        self.recovery_df = None


    def recovery_summary(self) -> pd.DataFrame:
        recovery_df = self.build_recovery_table()
        summary = []

        for model in sorted(recovery_df["model"].unique()):
            df_model = recovery_df[recovery_df["model"] == model]
            total = len(df_model)
            exact = (df_model["recovery"] == "Exact").sum()
            approximate = (df_model["recovery"] == "Approximate").sum()
            failed = (df_model["recovery"] == "Failed").sum()

            exact_rate = (100 * exact / total if total > 0 else 0.0)

            overall_rate = (100 * (exact + approximate) / total if total > 0 else 0.0)

            summary.append(
                {
                    "model": model,
                    "total": total,
                    "exact": exact,
                    "approximate": approximate,
                    "failed": failed,
                    "exact_recovery_rate (%)": round(exact_rate, 2),
                    "overall_recovery_rate (%)": round(overall_rate, 2),
                }
            )

        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values(
            by=[
                "overall_recovery_rate (%)",
                "exact_recovery_rate (%)",
            ],
            ascending=False,
        ).reset_index(drop=True)

        return summary_df
    

    def benchmark_recovery_matrix(self) -> pd.DataFrame:
        recovery_df = self.build_recovery_table()
        matrix = recovery_df.pivot(index="benchmark", columns="model", values="recovery")

        model_order = [
            c for c in self.df["model"].unique()
            if c in matrix.columns
        ]
        matrix = matrix[model_order]
        matrix = matrix.sort_index()
        return matrix


    #export
    # ---------------------------------------------------------
    def export_latex(self, table: pd.DataFrame, filename: str, caption: str = "Equation Recovery Results", label: str = "tab:equation_recovery"):
        latex = table.to_latex(
            index=False,
            escape=False,
            longtable=False,
            caption=caption,
            label=label,
            column_format="l" * len(table.columns),
        )
        with open(filename, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {filename}")


    def export_csv(self, table: pd.DataFrame, filename: str):
        table.to_csv(filename, index=False, encoding="utf-8")
        print(f"CSV table saved to {filename}")