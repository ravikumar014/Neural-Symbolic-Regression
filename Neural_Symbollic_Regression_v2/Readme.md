# ANSR v2 — Adaptive Neural Symbolic Regression

**From neural approximation to interpretable mathematical equations.**

Welcome to **ANSR v2**, the next step in our journey towards building efficient, reliable, and interpretable symbolic regression systems!

ANSR (Adaptive Neural Symbolic Regression) explores how deep learning and sparse mathematical modelling can work together to discover meaningful equations directly from data. Version 2 builds upon the foundation of v1, with a more refined framework for representation learning, equation recovery, and model evaluation.

The goal is simple: **make equation discovery more accurate, interpretable, and computationally efficient, without losing the flexibility of neural networks.**

## What's New in v2?

Version 1 established the core neural-symbolic pipeline, combining neural network approximation with sparse regression for mathematical equation discovery. With v2, the focus shifts towards improving the overall discovery process, making it more flexible, robust, and extensible.

| Feature               | ANSR v1                                                                     | ANSR v2                                                                        |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Neural representation | Interaction-aware neural approximation                                      | Enhanced representation learning and a more extensible discovery pipeline      |
| Equation recovery     | LASSO-based sparse extraction                                               | Refined sparse recovery and symbolic equation evaluation                       |
| Optimization          | Neural network hyperparameter tuning using Ray Tune and ASHA                | Improved scope for optimizing the equation discovery process                   |
| Search strategy       | Predetermined feature construction and extraction                           | Foundation for more adaptive and flexible search                               |
| Evaluation            | Prediction accuracy, noise robustness, symbolic recovery and generalization | Expanded evaluation and validation of recovered equations                      |
| Research direction    | Neural approximation followed by sparse recovery                            | Towards adaptive search, efficient extraction, and scalable equation discovery |

## Architecture Overview

ANSR v2 follows a neural-symbolic approach, where neural networks learn the underlying relationships in the data and symbolic methods recover compact mathematical expressions.

### 1. Data Processing and Representation Learning

The input dataset is processed and transformed into a suitable representation for learning. The neural component captures nonlinear relationships and interactions between input variables, providing a learned approximation of the underlying function.

### 2. Neural Function Approximation

A neural network learns a smooth approximation of the target function. This helps capture complex relationships and provides a useful foundation for subsequent symbolic recovery, particularly when working with noisy observations.

### 3. Symbolic Equation Recovery

The learned representation is used to support the recovery of mathematical expressions through sparse regression and symbolic modelling. The objective is to identify a compact set of meaningful terms while maintaining predictive accuracy.

### 4. Equation Evaluation and Validation

Recovered expressions are evaluated for numerical accuracy, symbolic structure, complexity, and generalization. This provides a more complete picture of the quality of a discovered equation, rather than relying on prediction error alone.

## Why v2?

The central motivation behind v2 is to move beyond simply fitting neural networks and extracting equations. We want to make the entire discovery process more intelligent and efficient.

Some of the key improvements and research objectives include:

* **Improved representation learning:** Better capture nonlinear interactions and provide useful representations for symbolic recovery.
* **More reliable equation extraction:** Focus on obtaining interpretable and mathematically meaningful expressions, rather than just numerical approximations.
* **Improved evaluation:** Consider predictive performance, equation complexity, symbolic fidelity, and generalization together.
* **Extensible architecture:** Establish a foundation for incorporating more advanced search and optimization strategies.

## Future Work

ANSR is an evolving research project, and there are several exciting directions we aim to explore next.

**1. Adaptive Symbolic Search**

Develop a search mechanism that dynamically adapts its mathematical operators, candidate expressions, and search space based on the dataset and previously discovered equations.

**2. Efficient Equation Extraction**

Investigate more efficient sparse recovery and symbolic simplification strategies to reduce computational overhead while preserving equation accuracy and interpretability.

**3. Search-History Guided Learning**

Explore how information from previously evaluated candidate equations can guide subsequent searches, reducing redundant computation and improving the discovery process.

**4. Budget-Aware Equation Discovery**

Introduce computationally aware search strategies that balance equation complexity, prediction accuracy, and available computational resources.

**5. Broader Scientific Applications**

Extend the framework to more challenging mathematical benchmarks and real-world scientific datasets, with an emphasis on discovering interpretable governing relationships.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/ravikumar014/Neural-Symbolic-Regression.git
cd Neural-Symbolic-Regression/Neural_Symbollic_Regression_v2
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Refer to the individual scripts and configuration files in the repository for the available experiments and execution instructions.

## Repository Structure

```text
Neural_Symbollic_Regression_v2/
│
├── Configuration and model settings
├── Neural representation learning
├── Symbolic equation recovery
├── Evaluation and experimentation
├── Benchmarking and results
└── Supporting utilities
```

*The structure above is a conceptual overview; refer to the actual repository files for the current implementation.*

## Research & Development

ANSR v2 is part of an ongoing effort to explore the intersection of deep learning, sparse modelling, and symbolic equation discovery.

The initial research explores neural networks as functional preconditioners for symbolic discovery, combining interaction-aware representations, sparse regression, and hyperparameter optimization. The accompanying research preprint is available here:

**[Neural Symbolic Regression Using Deep Learning and Sparse Modelling — arXiv](https://arxiv.org/abs/2609.01102)**

## Contributing

This is an evolving research project, and contributions, suggestions, experiments, and constructive discussions are always welcome!

Whether you're interested in neural networks, symbolic regression, mathematical optimization, or scientific machine learning, feel free to explore the repository, experiment with the framework, and share your ideas.

The long-term vision is to build a symbolic regression framework that doesn't just discover equations, but learns to discover them **smarter, faster, and more reliably.**

**Let's make mathematical discovery more accessible, interpretable, and efficient!**
