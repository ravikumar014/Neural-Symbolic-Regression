# Neural Symbolic Regression

> **Building more efficient, reliable, and interpretable ways to discover mathematical equations from data.**

Neural Symbolic Regression (NSR) is an ongoing research project exploring how **neural networks, sparse modelling, and symbolic search** can work together to discover mathematical relationships hidden inside data.

The project started with a simple question:

**Can we use neural representations to make symbolic regression more efficient without losing interpretability?**

The first version explores this through a neural functional representation followed by sparse equation recovery using techniques such as **LASSO/SINDy**. The current implementation, experiments, ablations, and results are available in the [`Neural-Symbollic-Regression-v1-main`](./Neural-Symbollic-Regression-v1-main) directory.

---

##  Where we are heading

V1 is only the starting point.

The broader goal is to move beyond fixed symbolic search spaces and develop a system that can **adapt its search strategy to the problem itself**.

Some of the directions we are actively exploring include:

* **Adaptive symbolic grammars** — dynamically deciding which operators and structures are useful.
* **Search-space routing** — avoiding unnecessary regions of the symbolic search space.
* **Learning from search history** — using previous discoveries to guide future searches.
* **Budget-aware symbolic regression** — making the search computationally efficient and scalable.
* **Robust symbolic discovery** — improving reliability under noise, distribution shifts, and limited data.
* **Hybrid neural + symbolic methods** — combining representation learning with explicit mathematical reasoning.

The long-term vision is a symbolic regression framework that doesn't blindly search through enormous expression spaces, but instead **learns how to search efficiently**.

---

## 🛠️ Current Work

The repository will evolve through multiple versions and experiments.

```text
Neural Symbolic Regression
│
├── V1 — Neural representation + sparse equation recovery
│
├── V2 — Improved symbolic discovery
│
├── Adaptive search
│
├── Search-space routing
│
├── Search-history learning
│
└── Efficient / budget-aware symbolic regression
```

Each stage is intended to be independently reproducible while contributing toward the larger research direction.

---

## 🤝 A Collaborative Research Project

This repository is intentionally being developed as an **open, collaborative research project**.

Ideas, experiments, alternative approaches, implementations, benchmarks, failures, and improvements are all welcome.

Not everything here will work — and that's part of the point.

The goal is to experiment, measure, learn, and gradually build a better symbolic regression system together.

**If you're interested in symbolic regression, scientific machine learning, neural-symbolic AI, mathematical discovery, or efficient search, feel free to contribute.**

---

## Research Philosophy

> **Don't just make symbolic regression more powerful.
> Make it smarter about where and how it searches.**

This project is a work in progress. Expect experiments, failed ideas, new directions, and several iterations along the way.

**Let's see how far we can push neural-symbolic mathematical discovery.** 🚀
