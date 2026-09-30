# ANSR v2 — Neural Symbolic Regression

**Learning complex functions with neural networks, then turning them back into equations.**

ANSR v2 is the second iteration of my Neural Symbolic Regression framework, focused on making symbolic equation discovery more accurate, interpretable, and computationally practical.

The main idea is to combine the representation learning capability of neural networks with the interpretability of sparse symbolic models. Instead of treating symbolic regression purely as a brute-force search problem, ANSR uses a learned representation as a starting point for recovering compact mathematical expressions.

> **Neural learning → structured representation → sparse recovery → interpretable equation**

---

## What changed from v1?

Version 1 established the basic neural-symbolic pipeline. With v2, the focus has shifted towards making the discovery pipeline more robust and easier to extend towards adaptive symbolic search.

### Key upgrades

|                    | ANSR v1                                          | ANSR v2                                                         |
| ------------------ | ------------------------------------------------ | --------------------------------------------------------------- |
| Representation     | Neural approximation of the target function      | Improved representation learning and symbolic recovery pipeline |
| Equation recovery  | Sparse regression over generated representations | More structured and extensible symbolic recovery                |
| Evaluation         | Prediction-focused evaluation                    | Prediction, equation structure, complexity and validation       |
| Search             | Relatively fixed discovery process               | Designed to support adaptive search strategies                  |
| Optimization       | Model/hyperparameter optimisation                | More emphasis on efficient discovery and computational budget   |
| Research direction | Neural approximation + sparse recovery           | Adaptive search + efficient equation extraction                 |

The goal of v2 is therefore not just to make the neural model stronger. It is to make the **whole equation discovery process smarter and more efficient**.

---

## Benchmark Performance

ANSR was evaluated against established symbolic regression baselines including **PySR** and **GPLearn** on standard symbolic regression benchmark families.

The benchmark suite currently covers:

* Nguyen
* Keijzer
* Vladislavleva

Across the reported experiments, all **24 runs completed successfully**.

### Overall benchmark summary

| Model        |       RMSE |        MAE |         R² | Train Time | Expression Complexity |
| ------------ | ---------: | ---------: | ---------: | ---------: | --------------------: |
| **NeuralSR** | **0.0306** | **0.0198** | **0.9915** | **3.08 s** |              **8.88** |
| PySR         |    0.00337 |    0.00243 |    0.99935 |    36.99 s |                 16.12 |
| GPLearn      |    0.08267 |    0.05842 |    0.84938 |    14.59 s |                 17.25 |

*Aggregated over 8 benchmark runs per model.*

A particularly interesting observation is the difference in computational cost. In this benchmark run, NeuralSR completed training in roughly **3 seconds on average**, compared with roughly **37 seconds for PySR**. At the same time, NeuralSR produced substantially lower average expression complexity than both comparison methods.

This highlights one of the main motivations behind ANSR: symbolic regression does not necessarily need to rely on an expensive search over a huge expression space.

---

## Benchmark Examples

### Nguyen

On the Nguyen benchmark family, NeuralSR consistently achieved high predictive accuracy.

For example, on Nguyen-1:

```text
RMSE : 0.018911
MAE  : 0.015133
R²   : 0.999568
Time : 5.950 s
```

On Nguyen-3:

```text
RMSE : 0.020532
MAE  : 0.013934
R²   : 0.999709
Time : 3.620 s
```

These results show that the learned representation can capture highly nonlinear relationships while still allowing a symbolic expression to be recovered.

### Keijzer

On Keijzer-1:

```text
RMSE : 0.009475
MAE  : 0.007303
R²   : 0.992776
Time : 3.124 s
```

and on Keijzer-2:

```text
RMSE : 0.013379
MAE  : 0.006025
R²   : 0.996146
Time : 2.228 s
```

The model also produced relatively compact expressions with only 5 active terms on both examples.

### Vladislavleva

The more challenging multivariate benchmarks provide another useful test.

For Vladislavleva-1:

```text
RMSE : 0.033256
MAE  : 0.016855
R²   : 0.966024
Time : 3.432 s
```

and Vladislavleva-4:

```text
RMSE : 0.054053
MAE  : 0.039785
R²   : 0.982490
Time : 2.406 s
```

## These experiments are particularly useful because they move beyond simple one-variable symbolic functions and test the approach on multivariate nonlinear relationships.

## Real-World Validation

Beyond synthetic symbolic regression benchmarks, the framework was also tested on real-world regression data.

For the larger validation experiment, the dataset contained **20,640 samples and 8 features**. The recovered symbolic model achieved:

```text
RMSE : 0.5363
MAE  : 0.3700
R²   : 0.7805
```

with **28 active symbolic terms**.

Statistical diagnostics were also performed on the recovered model, including F-test, Shapiro-Wilk, Breusch-Pagan and Durbin-Watson statistics.

This part of the work is important because the objective is not only to recover equations from clean synthetic functions, but eventually to discover useful interpretable relationships from real datasets.

---

## Why ANSR?

Traditional symbolic regression methods can require searching through a very large combinatorial space of operators, variables and expression structures.

ANSR explores a different direction:

**Learn first. Search smarter. Extract an equation.**

The neural component provides a flexible representation of the underlying function, while sparse symbolic recovery pushes the solution back towards an interpretable mathematical form.

This creates an interesting trade-off between:

* predictive accuracy
* symbolic complexity
* computational cost
* interpretability
* generalisation

---

## Future Work

The next stage of ANSR is moving towards a more adaptive symbolic discovery system.

### Adaptive Search

Instead of using the same symbolic search space for every problem, future versions will investigate dynamically selecting operators and expression structures based on the data and previously discovered candidates.

### Search-Space Routing

Different problems may require very different mathematical structures. A future ANSR system could learn which parts of the symbolic search space are actually useful and avoid spending computation on irrelevant expressions.

### Search-History Learning

Previously evaluated equations contain useful information. Future versions will explore using search history to guide subsequent candidate generation and reduce repeated exploration.

### Efficient Equation Extraction

Another major direction is reducing the cost of converting learned representations into clean symbolic equations, while maintaining accuracy and mathematical simplicity.

### Budget-Aware Symbolic Regression

Ultimately, the aim is to make symbolic regression aware of its computational budget and dynamically balance:

```text
Accuracy  ↔  Complexity  ↔  Search Cost
```

The long-term goal is an **adaptive symbolic regression system that learns not only the function, but also how to search for its equation efficiently.**

---

## Research Direction

ANSR is still an active research project, and v2 is intended as a foundation for further experimentation rather than a final system.

The broader direction is to investigate how **deep representation learning, sparse modelling and adaptive symbolic search** can work together for interpretable scientific machine learning.

Research preprint:

**Neural Symbolic Regression Using Deep Learning and Sparse Modeling**

https://arxiv.org/abs/2609.01102

---

## Repository

```bash
git clone https://github.com/ravikumar014/Neural-Symbolic-Regression.git

cd Neural-Symbolic-Regression/Neural_Symbollic_Regression_v2
```

The repository contains the implementation, benchmark utilities, evaluation code and experiments for the ANSR framework.

---

## What's next?

ANSR v2 is basically a step towards a bigger question:

> **Can a model learn how to search for mathematical equations, instead of blindly searching the entire symbolic space?**

That is where the next versions are heading.
