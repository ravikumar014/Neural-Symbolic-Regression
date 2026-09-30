"""
benchmark_runner.py

Runs all symbolic regression benchmarks using the Neural Symbolic
Regression framework.

Author: Ravi Kumar U
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from benchmarks import Benchmarks

from nsr import (
    LibraryConfig,
    ModelConfig,
    TrainingConfig,
    NeuralSRModel,
    train_val_split,
    rmse,
)

from symbollic_baselines import PySRBaseline, GPLearnBaseline
from configs import PySRConfig, GPLearnConfig
from statistical_significance import StatisticalSignificance
from equation_analysis import EquationAnalyzer


@dataclass
class BenchmarkResult:
    #model info
    model: str   # NSR, PySR, GPLearn
    family: str  # Nguyen, Keijzer, Vladislavleva
    benchmark: int

    #info of dataset
    n_variables: int
    n_samples: int

    #performance evaluation
    rmse: float
    mae: float
    r2: float

    #time
    train_time: float
    inference_time: float

    #complexity of the model
    expression_complexity: int
    n_library_features: int
    n_active_terms: int

    #symbolic expression
    true_equation: str
    extracted_equation: str

    #status
    success: bool
    error: str = ""


#benchmark runner main class
class BenchmarkRunner:
    def __init__(self, models, results_dir: str = "results"):
        self.models = models
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    # ---------------------------------------------------------
    def run_single_benchmark(self, family: str, benchmark_id: int, n_samples: int = 1000, noise: float = 0.0, random_seed: int = 1337) -> List[BenchmarkResult]:
        print("=" * 70)
        print(f"{family}-{benchmark_id}")
        print("=" * 70)

        #loading benchmark
        expr, variables = Benchmarks.get(family, benchmark_id)
        low, high = Benchmarks.default_domain(family)

        X, y = Benchmarks.sample(expr, variables, n_samples=n_samples, noise=noise, low=low, high=high, seed=random_seed)

        #train / val split
        X_train, y_train, X_val, y_val = train_val_split(X, y, val_split=0.2, seed=random_seed)
        benchmark_results = []

        #evaluating every model
        for model in self.models:
            print(f"\nEvaluating {model.get_model_name()}")
            try:
                #performance metrics
                metrics = self._evaluate_model(model=model, X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val, X_val=X_val, y_val=y_val)

                # model metadata
                info = self._collect_metadata(model)

                print(f"RMSE       : {metrics['rmse']:.6f}")
                print(f"MAE        : {metrics['mae']:.6f}")
                print(f"R²         : {metrics['r2']:.6f}")
                print(f"Train Time : {metrics['train_time']:.3f}s")
                print(f"Model      : {info['model']}")

                benchmark_results.append(
                    BenchmarkResult(
                        model=info["model"],
                        family=family,
                        benchmark=benchmark_id,
                        n_variables=len(variables),
                        n_samples=n_samples,
                        rmse=metrics["rmse"],
                        mae=metrics["mae"],
                        r2=metrics["r2"],
                        train_time=metrics["train_time"],
                        inference_time=metrics["inference_time"],
                        expression_complexity=info["expression_complexity"],
                        n_library_features=info["library_size"],
                        n_active_terms=info["active_terms"],
                        true_equation=str(expr),
                        extracted_equation=info["equation"],
                        success=True,
                    )
                )

            except Exception as e:
                print(f"{model.get_model_name()} FAILED")
                print(e)

                benchmark_results.append(
                    BenchmarkResult(
                        model=model.get_model_name(),
                        family=family,
                        benchmark=benchmark_id,
                        n_variables=len(variables),
                        n_samples=n_samples,
                        rmse=np.nan,
                        mae=np.nan,
                        r2=np.nan,
                        train_time=np.nan,
                        inference_time=np.nan,
                        expression_complexity=-1,
                        n_library_features=-1,
                        n_active_terms=-1,
                        true_equation="",
                        extracted_equation="",
                        success=False,
                        error=str(e),
                    )
                )

        return benchmark_results

    # ---------------------------------------------------------
    def run_family(self, family: str):
        ids = Benchmarks.list_available()[family]

        print("\n")
        print("#" * 70)
        print(f"Running {family} Benchmarks")
        print("#" * 70)

        for benchmark in ids:
            benchmark_results = self.run_single_benchmark(family, benchmark)
            self.results.extend(benchmark_results)

    # ---------------------------------------------------------
    def run_all(self):
        print("\n")
        print("=" * 80)
        print("Neural Symbolic Regression Benchmark Suite")
        print("=" * 80)

        self.results.clear()

        for family in Benchmarks.list_available():
            self.run_family(family)
        print("\nCompleted all benchmarks.")


    #utility functions
    def results_to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(r) for r in self.results])
        if df.empty:
            return df

        column_order = [
            "model",
            "family",
            "benchmark",
            "n_variables",
            "n_samples",
            "rmse",
            "mae",
            "r2",
            "train_time",
            "inference_time",
            "expression_complexity",
            "n_library_features",
            "n_active_terms",
            "true_equation",
            "extracted_equation",
            "success",
            "error",
        ]

        df = df[column_order]

        return df.sort_values(
            by=["model", "family", "benchmark"],
            ignore_index=True,
        )


    # ---------------------------------------------------------
    def save_csv(self):
        path = self.results_dir / "benchmark_results.csv"
        self.results_to_dataframe().to_csv(path, index=False)
        print(f"Saved CSV    : {path}")


    # ---------------------------------------------------------
    def save_excel(self):
        path = self.results_dir / "benchmark_results.xlsx"
        self.results_to_dataframe().to_excel(path, index=False)
        print(f"Saved Excel  : {path}")


    # ---------------------------------------------------------
    def save_latex(self):
        path = self.results_dir / "benchmark_results.tex"
        df = self.results_to_dataframe()
        with open(path, "w") as f:
            f.write(
                df.to_latex(
                    index=False,
                    float_format="%.5f",
                    escape=False,
                )
            )
        print(f"Saved LaTeX  : {path}")


    # ---------------------------------------------------------
    def save_json(self):
        df = self.results_to_dataframe()
        path = self.results_dir / "summary.json"
        summary = {
            "overall": {
                "total_runs": len(df),
                "successful": int(df["success"].sum()),
                "failed": int((~df["success"]).sum()),
            },
            "models": {}
        }

        for model in sorted(df["model"].unique()):
            mdf = df[df["model"] == model]
            summary["models"][model] = {
                "runs": len(mdf),
                "successful": int(mdf["success"].sum()),
                "failed": int((~mdf["success"]).sum()),
                "average_rmse": float(mdf["rmse"].mean()),
                "average_mae": float(mdf["mae"].mean()),
                "average_r2": float(mdf["r2"].mean()),
                "average_train_time": float(mdf["train_time"].mean()),
                "average_inference_time": float(mdf["inference_time"].mean()),
                "average_expression_complexity": float(mdf["expression_complexity"].mean()),
                "average_active_terms": float(mdf["n_active_terms"].mean()),
            }
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved JSON   : {path}")


    # ---------------------------------------------------------
    # def export_results(self):
    #     self.save_csv()
    #     self.save_excel()
    #     self.save_latex()
    #     self.save_json()

    #     #equation analysis
    #     df = self.results_to_dataframe()

    #     analyzer = EquationAnalyzer(results_df=df, )
    #     recovery_table = analyzer.build_recovery_table()
    #     recovery_summary = analyzer.recovery_summary()
    #     recovery_matrix = analyzer.benchmark_recovery_matrix()
    #     analyzer.export_csv(recovery_table, self.results_dir / "equation_recovery.csv")
    #     analyzer.export_csv(recovery_summary, self.results_dir / "recovery_summary.csv")
    #     analyzer.export_csv(recovery_matrix, self.results_dir / "recovery_matrix.csv")
    #     analyzer.export_latex(recovery_table, self.results_dir / "equation_recovery.tex")
    #     analyzer.export_latex(recovery_summary, self.results_dir / "recovery_summary.tex")


    # ---------------------------------------------------------
    def print_summary(self):
        df = self.results_to_dataframe()

        print("\n")
        print("=" * 90)
        print("Benchmark Summary")
        print("=" * 90)

        print(df[
                [
                    "model",
                    "family",
                    "benchmark",
                    "rmse",
                    "mae",
                    "r2",
                    "train_time",
                    "expression_complexity",
                    "n_active_terms",
                    "success",
                ]
            ]
        )

        print("\n")
        print("=" * 90)
        print("Overall")
        print("=" * 90)

        print(f"Total Runs : {len(df)}")
        print(f"Successful : {df['success'].sum()}")
        print(f"Failed     : {(~df['success']).sum()}")

        print("\n")
        print("=" * 90)
        print("Model-wise Performance")
        print("=" * 90)

        for model in sorted(df["model"].unique()):
            mdf = df[df["model"] == model]
            print(f"\n{model}")
            print("-" * 90)
            print(f"Runs                : {len(mdf)}")
            print(f"RMSE                : {mdf['rmse'].mean():.6f}")
            print(f"MAE                 : {mdf['mae'].mean():.6f}")
            print(f"R²                  : {mdf['r2'].mean():.6f}")
            print(f"Train Time          : {mdf['train_time'].mean():.3f} sec")
            print(f"Inference Time      : {mdf['inference_time'].mean():.6f} sec")
            print(f"Expression Complexity : {mdf['expression_complexity'].mean():.2f}")
            print(f"Active Terms        : {mdf['n_active_terms'].mean():.2f}")
            print(f"Success Rate        : {mdf['success'].mean()*100:.1f}%")

        print("=" * 90)


    # ---------------------------------------------------------
    def _evaluate_model(self, model, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
        #training
        start = time.perf_counter()

        if X_val is None or y_val is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, X_val, y_val,)

        train_time = time.perf_counter() - start

        #prediction
        start = time.perf_counter()
        pred = model.predict(X_val)
        inference_time = time.perf_counter() - start

        #metrics
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        return {
            "train_time": train_time,
            "inference_time": inference_time,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }


    # ---------------------------------------------------------
    # def _collect_metadata(self, model):
    #     return {
    #         "model": model.get_model_name(),
    #         "equation": model.get_equation(),
    #         "active_terms": model.get_active_terms(),
    #         "library_size": model.get_library_size(),
    #         "expression_complexity":model.get_expression_complexity(),
    #     }
    def _collect_metadata(self, model):
        eq = model.get_equation()

        print(f"\n{model.get_model_name()}")
        print(eq)
        print("-" * 60)

        return {
            "model": model.get_model_name(),
            "equation": eq,
            "active_terms": model.get_active_terms(),
            "library_size": model.get_library_size(),
            "expression_complexity": model.get_expression_complexity(),
        }


#main function
def main():
    np.random.seed(1337)

    library_cfg = LibraryConfig(
        funcs=[
            "id",
            "sin",
            "cos",
            "square",
            "cube",
            "sqrt",
            "inverse",
            "inverse_square",
            "abs",
            "exp",
            "tanh",
            "log1p",
        ],

        include_interactions=True,
        max_order=2,
        n_jobs=-1,
    )

    model_cfg = ModelConfig(
        hidden=128,
        depth=2,
        activation="relu",
        dropout=0.1,
        weight_decay=1e-4,
    )

    train_cfg = TrainingConfig(
        lr=1e-3,
        batch_size=128,
        epochs=200,
        scheduler="onecycle",
        early_stop_patience=20,
    )

    #models
    nsr_model = NeuralSRModel(
        library_cfg=library_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        extractor_alpha=1e-3,
    )

    pysr_cfg = PySRConfig()
    pysr_model = PySRBaseline(config=pysr_cfg)

    gp_cfg = GPLearnConfig()
    gplearn_model = GPLearnBaseline(config=gp_cfg)

    models = [nsr_model, pysr_model, gplearn_model]

    #benchmark runner
    runner = BenchmarkRunner(
        models=models,
        results_dir="results",
    )

    runner.run_all()
    # runner.export_results()
    runner.print_summary()


    #statistical significance test
    # from statistical_significance import StatisticalSignificance

    # stats = nsr_model.get_statistics()
    # results = stats.full_report()
    # print(results.summary)
    # print()
    # print(results.coefficient_table)
    # print()
    # print(results.shapiro)
    # print()
    # print(results.breusch_pagan)
    # print()
    # print(results.durbin_watson)
    # print()
    # print(results.vif_table)


if __name__ == "__main__":
    main()