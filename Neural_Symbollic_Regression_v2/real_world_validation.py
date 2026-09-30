import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from statistical_significance import StatisticalSignificance


class RealWorldValidation:
    def __init__(self, model, dataset_name="Unknown"):
        self.model = model
        self.dataset_name = dataset_name
        self.metrics = {}
        self.statistics = None


    def evaluate(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
        stats = StatisticalSignificance(self.model)
        self.statistics = stats.full_report()
        self.print_report(X, y)

        return self.metrics

    def print_report(self, X, y):
        print()
        print("=" * 70)
        print("REAL WORLD VALIDATION")
        print("=" * 70)
        print(f"Dataset          : {self.dataset_name}")
        print(f"Samples          : {len(y)}")
        print(f"Features         : {X.shape[1]}")
        print()
        print("-" * 70)
        print("Performance")
        print("-" * 70)
        print(f"RMSE             : {self.metrics['RMSE']:.6f}")
        print(f"MAE              : {self.metrics['MAE']:.6f}")
        print(f"R²               : {self.metrics['R2']:.6f}")
        print()
        print("-" * 70)
        print("Extracted Equation")
        print("-" * 70)
        print(self.model.symbolic_equation)
        print()
        print("-" * 70)
        print("Model Complexity")
        print("-" * 70)
        print(f"Active Terms     : {len(self.model.active_terms)}")
        print()
        print("-" * 70)
        print("Statistical Validation")
        print("-" * 70)
        print(f"F Statistic      : {self.statistics.f_statistic:.4f}")
        print(f"F p-value        : {self.statistics.f_pvalue:.4e}")
        print(f"Shapiro p-value  : {self.statistics.shapiro['p_value']:.4e}")
        print(f"BP p-value       : {self.statistics.breusch_pagan['LM p-value']:.4e}")
        print(f"Durbin-Watson    : {self.statistics.durbin_watson:.4f}")
        print()
        print("=" * 70)



from sklearn.datasets import load_diabetes, fetch_california_housing

from nsr import (
    LibraryConfig,
    ModelConfig,
    TrainingConfig,
    NeuralSRModel,
    train_val_split,
    rmse,
)
from real_world_validation import RealWorldValidation

# X, y = load_diabetes(return_X_y=True)
X, y = fetch_california_housing(return_X_y=True)

library_cfg = LibraryConfig(
    funcs=[
        "id",
        "sin",
        "cos",
        "square",
        "tanh",
        # "cube", 
        # "sqrt"
    ],
    include_interactions=False,
    max_order=1,
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

model = NeuralSRModel(
    library_cfg=library_cfg,
    model_cfg=model_cfg,
    train_cfg=train_cfg,
    extractor_alpha=1e-2,
)

validator = RealWorldValidation(model, dataset_name="Diabetes")
# validator = RealWorldValidation(model, dataset_name="California Housing")

validator.evaluate(X, y)