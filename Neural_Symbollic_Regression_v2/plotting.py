"""
plotting.py

Visualization utilities for Neural Symbolic Regression benchmark results.

Author: Ravi Kumar U
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# class BenchmarkPlotter:

#     def __init__(
#             self,
#             results_path="results/benchmark_results.csv",
#             figures_dir="figures",
#             dpi=300,
#         ):

#         self.results_path = Path(results_path)
#         self.figures_dir = Path(figures_dir)
#         self.figures_dir.mkdir(exist_ok=True)
#         self.dpi = dpi
#         self.df = None


#     # --------------------------------------------------------------
#     def load_results(self):
#         if not self.results_path.exists():
#             raise FileNotFoundError(f"{self.results_path} not found.")

#         self.df = pd.read_csv(self.results_path)

#         self.df["Benchmark"] = (
#             self.df["family"]
#             + "-"
#             + self.df["benchmark"].astype(str)
#         )

#         print(f"Loaded {len(self.df)} benchmark results.")


#     # --------------------------------------------------------------
#     def _save(self, fig, filename):
#         path = self.figures_dir / filename
#         fig.tight_layout()
#         fig.savefig(
#             path,
#             dpi=self.dpi,
#             bbox_inches="tight",
#         )
#         plt.close(fig)
#         print(f"Saved {path}")


#     # --------------------------------------------------------------
#     def plot_rmse(self):
#         fig, ax = plt.subplots(figsize=(10, 5))
#         bars = ax.bar(
#             self.df["Benchmark"],
#             self.df["rmse"],
#         )

#         ax.set_title("RMSE Across Benchmarks")
#         ax.set_ylabel("RMSE")
#         ax.set_xlabel("Benchmark")
#         ax.tick_params(axis="x", rotation=45)

#         for bar in bars:
#             h = bar.get_height()
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 h,
#                 f"{h:.3f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=8,
#                 rotation=90,
#             )

#         self._save(fig,"rmse.png")


#     # --------------------------------------------------------------
#     def plot_mae(self):
#         fig, ax = plt.subplots(figsize=(10, 5))
#         bars = ax.bar(
#             self.df["Benchmark"],
#             self.df["mae"],
#         )

#         ax.set_title("MAE Across Benchmarks")
#         ax.set_ylabel("MAE")
#         ax.set_xlabel("Benchmark")
#         ax.tick_params(axis="x", rotation=45)

#         for bar in bars:
#             h = bar.get_height()
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 h,
#                 f"{h:.3f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=8,
#                 rotation=90,
#             )

#         self._save(fig,"mae.png")


#     # --------------------------------------------------------------
#     def plot_r2(self):
#         fig, ax = plt.subplots(figsize=(10, 5))
#         bars = ax.bar(self.df["Benchmark"],
#                       self.df["r2"],
#                     )

#         ax.set_title("R² Across Benchmarks")
#         ax.set_ylabel("R² Score")
#         ax.set_xlabel("Benchmark")
#         ax.tick_params(axis="x", rotation=45)
#         ax.axhline(0, linestyle="--", linewidth=1)

#         for bar in bars:
#             h = bar.get_height()
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 h,
#                 f"{h:.3f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=8,
#                 rotation=90,
#             )

#         self._save(fig, "r2.png")


#     # --------------------------------------------------------------
#     def plot_training_time(self):
#         fig, ax = plt.subplots(figsize=(10, 5))
#         bars = ax.bar(
#             self.df["Benchmark"],
#             self.df["train_time"],
#         )

#         ax.set_title("Training Time per Benchmark")
#         ax.set_ylabel("Seconds")
#         ax.set_xlabel("Benchmark")
#         ax.tick_params(axis="x", rotation=45)

#         for bar in bars:
#             h = bar.get_height()
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 h,
#                 f"{h:.2f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=8,
#                 rotation=90,
#             )

#         self._save(fig, "training_time.png")


#     # --------------------------------------------------------------
#     def plot_active_terms(self):
#         fig, ax = plt.subplots(figsize=(10, 5))
#         bars = ax.bar(
#             self.df["Benchmark"],
#             self.df["n_active_terms"],
#         )
#         ax.set_title("Active Symbolic Terms")
#         ax.set_ylabel("Number of Active Terms")
#         ax.tick_params(axis="x", rotation=45)

#         for b in bars:
#             h = b.get_height()
#             ax.text(
#                 b.get_x()+b.get_width()/2,
#                 h,
#                 str(int(h)),
#                 ha="center",
#                 va="bottom",
#                 fontsize=8,
#             )
#         self._save(fig, "active_terms.png")


#     # --------------------------------------------------------------
#     def plot_family_summary(self):
#         summary = self.df.groupby("family").agg({
#             "rmse":"mean",
#             "mae":"mean",
#             "r2":"mean",
#             "train_time":"mean",
#         })

#         fig, ax = plt.subplots(figsize=(8,5))
#         summary.plot.bar(ax=ax)
#         ax.set_title("Average Performance by Benchmark Family")
#         ax.tick_params(axis="x", rotation=0)
#         self._save(fig,"family_summary.png")


#     # --------------------------------------------------------------
#     def plot_correlation_heatmap(self):
#         cols = ["rmse", "mae", "r2", "train_time", "n_active_terms"]
#         corr = self.df[cols].corr()
#         fig, ax = plt.subplots(figsize=(7,6))
#         im = ax.imshow(corr)
#         ax.set_xticks(range(len(cols)))
#         ax.set_yticks(range(len(cols)))
#         ax.set_xticklabels(cols, rotation=45)
#         ax.set_yticklabels(cols)

#         for i in range(len(cols)):
#             for j in range(len(cols)):
#                 ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)

#         fig.colorbar(im)
#         ax.set_title("Correlation Heatmap")
#         self._save(fig,"correlation_heatmap.png")


#     # --------------------------------------------------------------
#     def plot_success_matrix(self):
#         fig, ax = plt.subplots(figsize=(9,2))
#         values = self.df["success"].astype(int).values.reshape(1,-1)
#         im = ax.imshow(values, aspect="auto")
#         ax.set_yticks([])
#         ax.set_xticks(range(len(self.df)))
#         ax.set_xticklabels(self.df["Benchmark"], rotation=45,)
#         ax.set_title("Benchmark Success Matrix")

#         for i,v in enumerate(values[0]):
#             ax.text(i, 0, "✓" if v else "✗", ha="center", va="center", fontsize=14)

#         self._save(fig,"success_matrix.png")


#     # --------------------------------------------------------------
#     def plot_all(self):
#         print("\nGenerating Figures...\n")
#         self.plot_rmse()
#         self.plot_mae()
#         self.plot_r2()
#         self.plot_training_time()
#         self.plot_active_terms()
#         self.plot_family_summary()
#         self.plot_correlation_heatmap()
#         self.plot_success_matrix()
#         print("\nFinished generating all figures.")



# # Main
# def main():
#     plotter = BenchmarkPlotter()
#     plotter.load_results()
#     plotter.plot_all()


# if __name__ == "__main__":
#     main()


class BenchmarkPlotter:
    """
    Here we are generating figures for,
        • RMSE comparison
        • MAE comparison
        • R² comparison
        • Training time comparison
        • Expression complexity
        • Accuracy vs Complexity
        • Pareto front analysis
        • Benchmark family summary
        • Prediction vs Ground Truth (optional)

    Author
    ------
    Ravi Kumar U
    """

    def __init__(self, results_path="results/benchmark_results.csv", figures_dir="figures", dpi=300):
        self.results_path = Path(results_path)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)

        self.dpi = dpi
        self.df = None

        self.model_order = ["NeuralSR", "PySR", "GPLearn"]

        #color palette
        self.model_colors = {
            "NeuralSR": "#1f77b4",   # Blue
            "PySR": "#ff7f0e",       # Orange
            "GPLearn": "#2ca02c",    # Green
        }

        #markers for scatter/line plots
        self.model_markers = {
            "NeuralSR": "o",
            "PySR": "s",
            "GPLearn": "^",
        }

        #patterns
        self.model_hatches = {
            "NeuralSR": "",
            "PySR": "//",
            "GPLearn": "\\\\",
        }

        # Configure matplotlib once
        self.setup_plot_style()


    def setup_plot_style(self):
        plt.rcParams.update({
            #fonts
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],

            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,

            #figure size and quality
            "figure.figsize": (7, 4.5),
            "figure.dpi": self.dpi,

            #axes
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,

            #lines
            "lines.linewidth": 2,
            "lines.markersize": 7,

            #saving figusre
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",

            #vector graphics
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        })


    def _grouped_bar_plot(self, metric: str, ylabel: str, title: str, filename: str, higher_is_better: bool = False):
        benchmarks = self.df["Benchmark"].unique()

        x = np.arange(len(benchmarks))
        n_models = len(self.model_order)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, model in enumerate(self.model_order):
            model_df = (
                self.df[self.df["model"] == model]
                .set_index("Benchmark")
                .reindex(benchmarks)
            )

            values = model_df[metric].values
            positions = x + (i - (n_models - 1) / 2) * width

            ax.bar(
                positions,
                values,
                width=width,
                label=model,
                color=self.model_colors[model],
                edgecolor="black",
                linewidth=0.8,
                hatch=self.model_hatches[model],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=30, ha="right",)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Benchmark")
        ax.set_title(title)
        ax.legend(frameon=False, ncol=len(self.model_order), loc="upper center", bbox_to_anchor=(0.5, 1.12))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        self._save(fig, filename)


    def _scatter_plot(
            self,
            x_metric: str,
            y_metric: str,
            xlabel: str,
            ylabel: str,
            title: str,
            filename: str,
            annotate: bool = True,
        ):

        fig, ax = plt.subplots(figsize=(7, 5))
        for model in self.model_order:
            model_df = self.df[self.df["model"] == model]
            ax.scatter(
                model_df[x_metric],
                model_df[y_metric],
                s=80,
                marker=self.model_markers[model],
                color=self.model_colors[model],
                edgecolors="black",
                linewidth=0.8,
                alpha=0.9,
                label=model,
            )

            if annotate:
                for _, row in model_df.iterrows():
                    ax.annotate(
                        row["Benchmark"],
                        (
                            row[x_metric],
                            row[y_metric],
                        ),
                        xytext=(5, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        ax.set_axisbelow(True)
        self._save(fig, filename)


    def plot_family_summary(self, metric="rmse"):
        summary = (
            self.df
            .groupby(["family", "model"])[metric]
            .mean()
            .unstack()
            .reindex(columns=self.model_order)
        )

        families = summary.index.tolist()
        x = np.arange(len(families))
        width = 0.8 / len(self.model_order)
        fig, ax = plt.subplots(figsize=(8,5))

        for i, model in enumerate(self.model_order):
            values = summary[model].values
            positions = x + (i - (len(self.model_order)-1)/2) * width
            ax.bar(
                positions,
                values,
                width=width,
                color=self.model_colors[model],
                edgecolor="black",
                linewidth=0.8,
                hatch=self.model_hatches[model],
                label=model,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(families)
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("Benchmark Family")
        ax.set_title(f"Average {metric.upper()} by Benchmark Family")

        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5,1.12))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        self._save(fig, f"figure_family_{metric}")


    def plot_pareto_front(self):
        summary = (
            self.df
            .groupby("model")
            .agg({
                "rmse": "median",
                "expression_complexity": "median",
            })
            .reindex(self.model_order)
        )
        fig, ax = plt.subplots(figsize=(7, 5))

        for model in self.model_order:
            x = summary.loc[model, "expression_complexity"]
            y = summary.loc[model, "rmse"]
            ax.scatter(
                x,
                y,
                s=220,
                marker=self.model_markers[model],
                color=self.model_colors[model],
                edgecolors="black",
                linewidth=1.0,
                label=model,
                zorder=3,
            )

            ax.annotate(
                model,
                (x, y),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        ax.annotate(
            "Better",
            xy=(0.05, 0.08),
            xycoords="axes fraction",
            fontsize=10,
            color="green",
            fontweight="bold",
        )

        ax.arrow(
            0.18,
            0.18,
            -0.08,
            -0.08,
            transform=ax.transAxes,
            width=0.003,
            color="green",
            length_includes_head=True,
        )

        ax.set_xlabel("Median Expression Complexity")
        ax.set_ylabel("Median RMSE")
        ax.set_title("Accuracy vs Expression Complexity")
        ax.grid(linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        ax.set_axisbelow(True)
        self._save(fig, "figure_pareto_front")


    def load_results(self):
        if not self.results_path.exists():
            raise FileNotFoundError(f"{self.results_path} not found.")

        self.df = pd.read_csv(self.results_path)
        self.df["Benchmark"] = (self.df["family"] + "-" + self.df["benchmark"].astype(str))
        print(f"Loaded {len(self.df)} benchmark results.")

    
    def _save(self, fig, filename):
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            path = self.figures_dir / f"{filename}.{ext}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")

        plt.close(fig)
        print(f"Saved {filename} (png/pdf/svg)")


    def plot_rmse(self):
        self._grouped_bar_plot(
            metric="rmse",
            ylabel="RMSE",
            title="RMSE Comparison Across Benchmarks",
            filename="figure1_rmse",
        )


    def plot_mae(self):
        self._grouped_bar_plot(
            metric="mae",
            ylabel="MAE",
            title="MAE Comparison Across Benchmarks",
            filename="figure2_mae",
        )


    def plot_r2(self):
        self._grouped_bar_plot(
            metric="r2",
            ylabel="R² Score",
            title="R² Comparison Across Benchmarks",
            filename="figure3_r2",
            higher_is_better=True,
        )


    def plot_training_time(self):
        self._grouped_bar_plot(
            metric="train_time",
            ylabel="Training Time (s)",
            title="Training Time Comparison",
            filename="figure4_training_time",
        )


    def plot_complexity(self):
        self._grouped_bar_plot(
            metric="expression_complexity",
            ylabel="Expression Complexity",
            title="Expression Complexity Comparison",
            filename="figure5_complexity",
        )


    def plot_accuracy_vs_complexity(self):
        self._scatter_plot(
            x_metric="expression_complexity",
            y_metric="rmse",
            xlabel="Expression Complexity",
            ylabel="RMSE",
            title="Accuracy vs Expression Complexity",
            filename="figure_accuracy_vs_complexity",
    )
        

    def plot_training_vs_accuracy(self):
        self._scatter_plot(
            x_metric="train_time",
            y_metric="rmse",
            xlabel="Training Time (s)",
            ylabel="RMSE",
            title="Training Time vs Accuracy",
            filename="figure_training_vs_accuracy",
        )

    def plot_complexity_vs_time(self):
        self._scatter_plot(
            x_metric="expression_complexity",
            y_metric="train_time",
            xlabel="Expression Complexity",
            ylabel="Training Time (s)",
            title="Complexity vs Training Time",
            filename="figure_complexity_vs_time",
        )


    def plot_all(self):
        print("\nGenerating Figures...\n")
        self.plot_rmse()
        self.plot_mae()
        self.plot_r2()
        self.plot_training_time()
        self.plot_complexity()
        self.plot_accuracy_vs_complexity()
        self.plot_training_vs_accuracy()
        self.plot_complexity_vs_time()
        self.plot_family_summary()
        self.plot_pareto_front()
        print("\nFinished generating all figures.")


#main
def main():
    plotter = BenchmarkPlotter()
    plotter.load_results()
    plotter.plot_all()


if __name__ == "__main__":
    main()