from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import os

from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.regressionplots import influence_plot


#results container
@dataclass
class StatisticalResults:
    fitted_model: Optional[object] = None
    summary: Optional[object] = None
    r2: Optional[float] = None
    adjusted_r2: Optional[float] = None
    f_statistic: Optional[float] = None
    f_pvalue: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    n_samples: Optional[int] = None
    df_model: Optional[int] = None
    df_residual: Optional[int] = None
    residuals: Optional[object] = None
    predictions: Optional[object] = None
    coefficient_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    shapiro: Optional[dict] = None
    breusch_pagan: Optional[dict] = None
    durbin_watson: Optional[float] = None
    vif_table: pd.DataFrame = field(default_factory=pd.DataFrame)



#statistical significance
class StatisticalSignificance:
    def __init__(self, model):
        self.model = model
        self.results = StatisticalResults()
        self._load_from_model()
        self._validate_inputs()

    # ------------------------------------------------------
    def _load_from_model(self):
        self.X = self.model.X_train
        self.y = self.model.y_train
        self.Phi = self.model.Phi_train
        self.feature_names = self.model.feature_names
        self.coefficients = self.model.coefficients
        self.symbolic_equation = self.model.symbolic_equation
        self.active_terms = self.model.active_terms

    # ------------------------------------------------------
    def _validate_inputs(self):
        required = {
            "X_train": self.X,
            "y_train": self.y,
            "Phi_train": self.Phi,
            "feature_names": self.feature_names,
            "coefficients": self.coefficients,
        }

        missing = []

        for name, value in required.items():
            if value is None:
                missing.append(name)

        if len(missing) > 0:
            raise RuntimeError(
                "Model is incomplete.\n"
                "Missing attributes:\n"
                + "\n".join(missing)
                + "\n\n"
                "Did you call model.fit() ?"
            )

        if self.Phi.shape[1] != len(self.feature_names):
            raise ValueError("Feature names do not match feature library.")

        if self.Phi.shape[1] != len(self.coefficients):
            raise ValueError("Coefficient vector size mismatch.")

    # ------------------------------------------------------
    def info(self):
        print()
        print("=" * 70)
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print("=" * 70)
        print(f"Samples              : {self.X.shape[0]}")
        print(f"Input Variables      : {self.X.shape[1]}")
        print(f"Library Size         : {self.Phi.shape[1]}")
        print(f"Active Terms         : {len(self.active_terms)}")
        print()
        print("Extracted Equation")
        print("------------------")
        print(self.symbolic_equation)
        print("=" * 70)

    # ------------------------------------------------------
    def get_active_library(self):
        mask = np.abs(self.coefficients) > 1e-8
        Phi_active = self.Phi[:, mask]
        coef_active = self.coefficients[mask]
        feature_names_active = list(np.array(self.feature_names)[mask])
        return (Phi_active, coef_active, feature_names_active)

    # ------------------------------------------------------
    def preview_active_terms(self):
        _, coef, names = self.get_active_library()
        print()
        print("=" * 70)
        print("ACTIVE SYMBOLIC TERMS")
        print("=" * 70)
        for c, n in zip(coef, names):
            print(f"{c:12.6f}    {n}")
        print("=" * 70)


    def fit(self):
        Phi_active, coef_active, feature_names = self.get_active_library()

        if Phi_active.shape[1] == 0:
            raise RuntimeError(
                "No active symbolic terms were selected by NSR."
            )
        Phi_active = sm.add_constant(Phi_active)

        ols_model = sm.OLS(
            self.y,
            Phi_active
        )

        results = ols_model.fit()
        self.results.fitted_model = results
        self.results.summary = results.summary()
        self.results.residuals = results.resid
        self.active_feature_names = feature_names
        self.active_coefficients = coef_active
        self._populate_model_statistics()

        return results


    def coefficient_statistics(self):
        if self.results.fitted_model is None:
            raise RuntimeError("Call fit() before coefficient_statistics().")
        results = self.results.fitted_model
        names = list(results.model.exog_names)
        ci = results.conf_int()
        table = pd.DataFrame({
            "Term": names,
            "Coefficient": np.asarray(results.params),
            "Std_Error": np.asarray(results.bse),
            "t_statistic": np.asarray(results.tvalues),
            "p_value": np.asarray(results.pvalues),
            "CI_Lower": ci[:, 0],
            "CI_Upper": ci[:, 1]
        })
        table["Significant"] = table["p_value"] < 0.05
        self.results.coefficient_table = table
        return table

    
    #helper functions
    def print_coefficient_table(self):
        table = self.coefficient_statistics()
        print()
        print("=" * 90)
        print("COEFFICIENT SIGNIFICANCE")
        print("=" * 90)
        print(table.to_string(index=False))
        print("=" * 90)

    def model_statistics(self):
        if self.results.fitted_model is None:
            raise RuntimeError("Call fit() first.")

        r = self.results.fitted_model
        summary = {
            "R2": r.rsquared,
            "Adjusted_R2": r.rsquared_adj,
            "F_statistic": r.fvalue,
            "F_pvalue": r.f_pvalue,
            "AIC": r.aic,
            "BIC": r.bic,
            "Num_Samples": int(r.nobs),
            "Degrees_Freedom": int(r.df_model),
            "Residual_DOF": int(r.df_resid)
        }
        return summary
    

    def _populate_model_statistics(self):
        r = self.results.fitted_model
        self.results.r2 = r.rsquared
        self.results.adjusted_r2 = r.rsquared_adj
        self.results.f_statistic = r.fvalue
        self.results.f_pvalue = r.f_pvalue
        self.results.aic = r.aic
        self.results.bic = r.bic
        self.results.n_samples = int(r.nobs)
        self.results.df_model = int(r.df_model)
        self.results.df_residual = int(r.df_resid)
    
    def residual_analysis(self):
        if self.results.fitted_model is None:
            raise RuntimeError("Call fit() first.")

        residuals = self.results.fitted_model.resid
        fitted = self.results.fitted_model.fittedvalues
        self.results.residuals = residuals
        self.results.predictions = fitted
        return residuals
    
    def shapiro_test(self):
        stat, p = shapiro(self.results.residuals)
        self.results.shapiro = {
            "Statistic": stat,
            "p_value": p,
            "Normal": p > 0.05
        }
        return self.results.shapiro
    
    def breusch_pagan_test(self):
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(self.results.residuals, self.results.fitted_model.model.exog)
        self.results.breusch_pagan = {
            "LM Statistic": lm,
            "LM p-value": lm_pvalue,
            "F Statistic": fvalue,
            "F p-value": f_pvalue,
            "Homoscedastic": lm_pvalue > 0.05
        }
        return self.results.breusch_pagan
    
    def durbin_watson_test(self):
        dw = durbin_watson(self.results.residuals)
        self.results.durbin_watson = dw
        return dw
    
    def vif(self):
        X = self.results.fitted_model.model.exog
        columns = ["Intercept"] + self.active_feature_names
        vif = []
        for i in range(X.shape[1]):
            vif.append(variance_inflation_factor(X, i))

        table = pd.DataFrame({
            "Feature": columns,
            "VIF": vif
        })
        self.results.vif_table = table
        return table
    
    def full_report(self):
        self.fit()
        self.coefficient_statistics()
        self.residual_analysis()
        self.shapiro_test()
        self.breusch_pagan_test()
        self.durbin_watson_test()
        self.vif()
        self.plot_residuals_vs_fitted()
        self.plot_residual_histogram()
        self.qq_plot()
        self.prediction_plot()
        self.cooks_distance()
        return self.results
    
    
    def _save_plot(self, filename):
        output_dir = "figures"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

    
    #residual vs fitted
    def plot_residuals_vs_fitted(self):
        fitted = self.results.predictions
        residuals = self.results.residuals
        plt.figure(figsize=(6, 5))
        plt.scatter(fitted, residuals)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.tight_layout()
        self._save_plot("residuals_vs_fitted.png")


    #residual histogram
    def plot_residual_histogram(self):
        plt.figure(figsize=(6, 5))
        plt.hist(self.results.residuals, bins=30)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Residual Distribution")
        plt.tight_layout()
        self._save_plot("residual_histogram.png")

    
    #Q-Q plot
    def qq_plot(self):
        plt.figure(figsize=(6, 6))
        stats.probplot(self.results.residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        plt.tight_layout()
        self._save_plot("qq_plot.png")


    #actual vs predicted
    def prediction_plot(self):
        y = self.y
        pred = self.results.predictions
        plt.figure(figsize=(6, 6))
        plt.scatter(y, pred)
        mn = min(y.min(), pred.min())
        mx = max(y.max(), pred.max())
        plt.plot([mn, mx], [mn, mx], "--", linewidth=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        self._save_plot("actual_vs_predicted.png")


    #cooks distance
    def cooks_distance(self):
        plt.figure(figsize=(8, 6))
        influence_plot(self.results.fitted_model)
        plt.title("Influence Plot")
        plt.tight_layout()
        self._save_plot("cooks_distance.png")