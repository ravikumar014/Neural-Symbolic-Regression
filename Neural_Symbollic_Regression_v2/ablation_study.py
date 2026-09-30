"""
ablation_study.py

Ablation study for the proposed Neural Symbolic Regression (NSR) model.
here we are evaluating the contribution of major architectural choices:
1. Symbolic function library
2. Interaction terms
3. Higher-order feature generation
4. Sparse equation recovery method
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from benchmarks import Benchmarks
from nsr import NeuralSRModel

from configs import (
    LibraryConfig,
    ModelConfig,
    TrainingConfig,
)

@dataclass
class AblationExperiment:
    name: str
    library_cfg: LibraryConfig
    model_cfg: ModelConfig
    train_cfg: TrainingConfig
    extractor_method: str = "lasso"


@dataclass
class AblationResult:
    experiment: str
    benchmark: str
    rmse: float
    mae: float
    r2: float
    train_time: float
    inference_time: float
    library_size: int
    active_terms: int
    expression_complexity: int
    equation: str



class AblationStudy:
    def __init__(self):
        self.experiments: List[AblationExperiment] = []
        self.results: List[AblationResult] = []

    def add_experiment(self, experiment: AblationExperiment):
        self.experiments.append(experiment)

    def create_default_experiments(self, library_cfg, model_cfg, train_cfg):
        self.add_experiment(
            AblationExperiment(
                name="Full NSR",
                library_cfg=copy.deepcopy(library_cfg),
                model_cfg=copy.deepcopy(model_cfg),
                train_cfg=copy.deepcopy(train_cfg),
                extractor_method="lasso",
            )
        )


        reduced = copy.deepcopy(library_cfg)
        reduced.funcs = ["id", "sin", "cos"]
        self.add_experiment(
            AblationExperiment(
                name="Reduced Function Library",
                library_cfg=reduced,
                model_cfg=copy.deepcopy(model_cfg),
                train_cfg=copy.deepcopy(train_cfg),
                extractor_method="lasso",
            )
        )


        no_interactions = copy.deepcopy(library_cfg)
        no_interactions.include_interactions = False
        self.add_experiment(
            AblationExperiment(
                name="No Interaction Terms",
                library_cfg=no_interactions,
                model_cfg=copy.deepcopy(model_cfg),
                train_cfg=copy.deepcopy(train_cfg),
                extractor_method="lasso",
            )
        )


        first_order = copy.deepcopy(library_cfg)
        first_order.max_order = 1
        self.add_experiment(
            AblationExperiment(
                name="First Order Library",
                library_cfg=first_order,
                model_cfg=copy.deepcopy(model_cfg),
                train_cfg=copy.deepcopy(train_cfg),
                extractor_method="lasso",
            )
        )


        self.add_experiment(
            AblationExperiment(
                name="ElasticNet Sparse Recovery",
                library_cfg=copy.deepcopy(library_cfg),
                model_cfg=copy.deepcopy(model_cfg),
                train_cfg=copy.deepcopy(train_cfg),
                extractor_method="elastic",
            )
        )


    def build_model(self, experiment: AblationExperiment):
        model = NeuralSRModel(
            library_cfg=experiment.library_cfg,
            model_cfg=experiment.model_cfg,
            train_cfg=experiment.train_cfg,
            extractor_alpha=1e-3,
            extractor_method=experiment.extractor_method,
        )
        return model
    
    def run_single_experiment(self, experiment: AblationExperiment, family, benchmark_id=1, n_samples=1000):
        expr, variables = Benchmarks.get(family, benchmark_id,)
        low, high = Benchmarks.default_domain(family)
        X, y = Benchmarks.sample(
            expr,
            variables,
            n_samples=n_samples,
            low=low,
            high=high,
            noise=0.0,
            seed=42,
        )

        model = self.build_model(experiment)
        start = time.perf_counter()
        model.fit(X, y)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        prediction = model.predict(X)
        inference_time = time.perf_counter() - start

        rmse = np.sqrt(mean_squared_error(y, prediction))
        mae = mean_absolute_error(y, prediction)
        r2 = r2_score(y, prediction)

        result = AblationResult(
            experiment=experiment.name,
            benchmark=f"{family}-{benchmark_id}",
            rmse=rmse,
            mae=mae,
            r2=r2,
            train_time=train_time,
            inference_time=inference_time,
            library_size=model.get_library_size(),
            active_terms=model.get_active_terms(),
            expression_complexity=model.get_expression_complexity(),
            equation=model.get_equation(),
        )
        self.results.append(result)
        print()
        print(
            f"{experiment.name} | {family}-{benchmark_id}"
        )
        print("=" * 80)
        print(f"RMSE      : {rmse:.6f}")
        print(f"MAE       : {mae:.6f}")
        print(f"R²        : {r2:.6f}")
        print(f"Library   : {model.get_library_size()}")
        print(f"Terms     : {model.get_active_terms()}")
        print(f"Time      : {train_time:.3f} sec")

    def run(self):
        available = Benchmarks.list_available()
        for experiment in self.experiments:
            print()
            print("=" * 100)
            print(f"Running Ablation : {experiment.name}")
            print("=" * 100)
            for family, benchmark_ids in available.items():
                print()
                print(f"Benchmark Family : {family}")
                for benchmark_id in benchmark_ids:
                    print(f"  -> {family}-{benchmark_id}")
                    self.run_single_experiment(
                        experiment=experiment,
                        family=family,
                        benchmark_id=benchmark_id,
                    )
        print()
        print("=" * 100)
        print("Finished All Ablation Experiments")
        print("=" * 100)

    def results_dataframe(self):
        rows = []
        for r in self.results:
            rows.append({
                "Experiment": r.experiment,
                "Benchmark": r.benchmark,
                "RMSE": r.rmse,
                "MAE": r.mae,
                "R2": r.r2,
                "Train_Time": r.train_time,
                "Inference_Time": r.inference_time,
                "Library_Size": r.library_size,
                "Active_Terms": r.active_terms,
                "Expression_Complexity": r.expression_complexity,
                "Equation": r.equation,
            })
        return pd.DataFrame(rows)
    
    def export_csv(self, filename="ablation_results.csv"):
        df = self.results_dataframe()
        df.to_csv(filename, index=False)
        print()
        print(f"Results saved to {filename}")


    def summary(self):
        print()
        print("=" * 80)
        print("Ablation Experiments")
        print("=" * 80)
        for i, exp in enumerate(self.experiments, start=1):
            print(f"{i}. {exp.name}")
        print()



if __name__ == "__main__":
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

    study = AblationStudy()
    study.create_default_experiments(library_cfg, model_cfg, train_cfg)
    study.summary()
    study.run()
    study.export_csv()