# Neural Symbolic Regression

> **Building more efficient, reliable, and interpretable ways to discover mathematical equations from data.**

Neural Symbolic Regression (NSR) is an ongoing research project exploring how **neural networks, sparse modelling, and symbolic search** can work together to discover mathematical relationships hidden inside data.

The project started with a simple question:

**Can we use neural representations to make symbolic regression more efficient without losing interpretability?**

What began as an exploration of neural functional representations and sparse equation recovery has now evolved into a broader effort to develop more adaptive, computationally efficient, and reliable symbolic discovery methods.

The repository currently includes two versions of the framework, each representing a step in this research journey.

---

## Versions

### V1 — Neural Representation + Sparse Equation Recovery

The first version established the core neural-symbolic pipeline, combining neural function approximation with sparse equation recovery using techniques such as **LASSO and SINDy**.

It explored how learned representations can support mathematical equation discovery, with experiments on standard symbolic regression benchmarks and real-world datasets.

[Explore V1](./Neural-Symbollic-Regression-v1-main)

### V2 — Improved Neural Symbolic Discovery

V2 builds upon the initial framework, with a stronger focus on representation learning, symbolic recovery, evaluation, and extensibility.

The objective is to improve the overall equation discovery process, balancing predictive accuracy, symbolic complexity, and computational efficiency.

The current benchmark experiments demonstrate promising results:

| Benchmark       |     R² |
| --------------- | -----: |
| Nguyen-3        | 0.9997 |
| Keijzer-2       | 0.9961 |
| Vladislavleva-4 | 0.9825 |

In the reported benchmark suite, the neural symbolic model achieved an average R² of **0.9915**, with an average training time of approximately **3.08 seconds** and expression complexity of **8.88**, across eight runs. These results were obtained in comparison with PySR and GPLearn.

The real-world validation experiments also explore equation recovery from datasets with multiple input features, extending the investigation beyond synthetic mathematical functions.

[Explore V2](./Neural_Symbollic_Regression_v2)

---

## Where We Are Heading

V1 established the foundation, and V2 advances the discovery pipeline. The broader goal is to move beyond fixed symbolic search spaces and develop a system that can **adapt its search strategy to the problem itself**.

Some of the research directions we are exploring include:

* **Adaptive symbolic grammars** — dynamically selecting useful mathematical operators and expression structures based on the problem.
* **Search-space routing** — identifying promising regions of the symbolic search space while avoiding unnecessary exploration.
* **Learning from search history** — using previously evaluated candidates and discovered equations to guide future searches.
* **Budget-aware symbolic regression** — balancing equation accuracy, complexity, and computational resources for more efficient discovery.
* **Robust symbolic discovery** — improving reliability under noise, distribution shifts, and limited data.
* **Hybrid neural + symbolic methods** — combining representation learning with explicit mathematical reasoning and structured equation recovery.
* **Efficient equation extraction** — developing faster and more reliable methods to transform learned representations into compact, interpretable mathematical expressions.

The long-term vision is a symbolic regression framework that doesn't blindly search through enormous expression spaces, but instead **learns how, where, and when to search efficiently**.

---

## Research Roadmap

The repository will continue to evolve through multiple versions, experiments, and research directions.

```text
Neural Symbolic Regression
│
├── V1 — Neural representation + sparse equation recovery
│
├── V2 — Improved symbolic discovery and evaluation
│
├── Adaptive symbolic grammars
│
├── Search-space routing
│
├── Search-history guided learning
│
├── Efficient equation extraction
│
└── Budget-aware and adaptive symbolic regression
```

Each stage is intended to be independently reproducible while contributing toward the larger research direction. Some of these are active research directions and planned extensions, rather than completed implementations.

---

## Research & Experiments

The project investigates symbolic discovery through a combination of:

* Neural function approximation and representation learning
* Sparse regression and mathematical equation recovery
* Symbolic regression benchmarking and comparative evaluation
* Ablation studies and statistical validation
* Real-world dataset experiments
* Computational efficiency and expression complexity analysis

The broader aim is to understand not just how accurately a model can approximate a function, but also **how effectively it can recover a meaningful mathematical expression**.

### Research Preprint

**Neural Symbolic Regression Using Deep Learning and Sparse Modeling**

[Read the preprint on arXiv](https://arxiv.org/abs/2609.01102)

---

## Repository Structure

```text
Neural-Symbolic-Regression/
│
├── Neural-Symbollic-Regression-v1-main/
│   └── V1: Neural representation and sparse recovery
│
├── Neural_Symbollic_Regression_v2/
│   └── V2: Improved neural symbolic discovery
│
└── README.md
```

Each version contains its own implementation, experiments, and supporting materials. Refer to the respective directories for setup instructions, execution details, and version-specific documentation.

---

## A Collaborative Research Project

This repository is intentionally being developed as an **open, collaborative research project**.

Ideas, experiments, alternative approaches, implementations, benchmarks, failures, and improvements are all welcome.

Not everything here will work — and that's part of the point.

The goal is to experiment, measure, learn, and gradually build a better symbolic regression system together.

If you're interested in symbolic regression, scientific machine learning, neural-symbolic AI, mathematical discovery, or efficient search, feel free to explore the repository, share ideas, or contribute.

---

## Research Philosophy

> **Don't just make symbolic regression more powerful. Make it smarter about where and how it searches.**

This project is a work in progress. Expect experiments, failed ideas, new directions, and several iterations along the way.

The journey from neural approximation to adaptive mathematical discovery is just getting started.

**Let's see how far we can push neural-symbolic mathematical discovery together!**
