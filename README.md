# Machine Learning Specialization — Self-Study Repository

This repository contains my personal implementations, notes, and assignments
from the **Machine Learning Specialization** by Andrew Ng (DeepLearning.AI &
Stanford Online). Everything here is written from scratch — no copy-pasted
solutions from the course notebooks — to make sure I actually understand the
math and the code behind each algorithm.

The goal is not just to finish the course, but to build a solid, reusable
reference of ML fundamentals implemented in plain NumPy, and later compared
against `scikit-learn` and other libraries.

---

## About the Course

The specialization consists of three courses:

1. **Supervised Machine Learning: Regression and Classification**
2. **Advanced Learning Algorithms**
3. **Unsupervised Learning, Recommenders, Reinforcement Learning**

It covers the theory and practice of the most common ML algorithms, with a
strong emphasis on:
- Building algorithms from scratch before using library implementations
- Vectorization with NumPy instead of explicit loops
- Understanding the cost functions and gradients behind every model
- Debugging models via learning curves, cost curves, and validation

---

## Topics Covered

### Course 1 — Supervised Learning: Regression and Classification
- Linear regression with one variable
- Cost function (MSE) and its intuition
- Gradient descent (batch, learning rate, convergence)
- Multiple linear regression
- Feature scaling and normalization
- Polynomial regression
- Logistic regression for binary classification
- Decision boundary, sigmoid function
- Regularization (L1, L2, Elastic Net)
- Overfitting, underfitting, bias-variance tradeoff

### Course 2 — Advanced Learning Algorithms
- Neural networks: forward propagation, architecture
- Activation functions (ReLU, sigmoid, softmax)
- Backpropagation and automatic differentiation
- Training neural networks in TensorFlow/Keras
- Multiclass classification
- Additional regularization techniques (dropout, early stopping)
- Model evaluation: precision, recall, F1, confusion matrix
- Decision trees, random forests, gradient boosting (XGBoost)

### Course 3 — Unsupervised Learning, Recommenders, RL
- K-means clustering
- Anomaly detection (Gaussian distribution)
- Dimensionality reduction with PCA
- Collaborative filtering and content-based recommenders
- Reinforcement learning: Markov Decision Processes, Q-learning

---

## What I Can Do After This Course

By working through the material and re-implementing everything myself, I am
able to:

- **Explain** what a cost function is, why we minimize it, and how to derive
  its gradient by hand.
- **Implement** linear and logistic regression, gradient descent, and
  regularization from scratch using only NumPy.
- **Vectorize** computations to avoid `for` loops and make code scale to
  large datasets.
- **Diagnose** model behavior using cost curves, learning curves, and
  validation metrics — and decide whether the problem is bias or variance.
- **Apply** feature engineering: scaling, normalization, polynomial features,
  interaction terms.
- **Build and train** neural networks in TensorFlow/Keras, including
  multiclass classification.
- **Use** decision trees, random forests, and boosted trees for tabular data.
- **Cluster** data with K-means, detect anomalies, and reduce dimensionality
  with PCA.
- **Build** simple recommender systems with collaborative filtering.
- **Understand** the basics of reinforcement learning and Q-learning.
- **Compare** my from-scratch implementations against `scikit-learn` to
  confirm correctness and understand trade-offs.

---

## Repository Structure

---

## Tools & Stack

- **Python 3**
- **NumPy** — all from-scratch implementations
- **Matplotlib** — visualizations and cost curves
- **scikit-learn** — for comparison and sanity checks
- **TensorFlow / Keras** — neural networks
- **Jupyter Notebook** — exploration and notes

---

## Notes to Myself (and Anyone Reading)

- No copy-paste from course notebooks. If a line of code is here, I wrote it
  and can explain it.
- Prefer vectorized NumPy over explicit loops — not for style, but because it
  matters on real datasets.
- Always check: does the cost go down? Is the learning rate too big or too
  small? Is the model overfitting?
- Every algorithm has a math derivation behind it. If I can't derive the
  gradient, I don't understand the algorithm yet.

---

## Progress

- [ ] Course 1 — Supervised Learning
  - [x] Week 1 — Linear regression, cost function
  - [ ] Week 2 — Gradient descent, multiple regression
  - [ ] Week 3 — Logistic regression, regularization
- [ ] Course 2 — Advanced Learning Algorithms
- [ ] Course 3 — Unsupervised Learning, Recommenders, RL

---

## License

Personal study materials. Feel free to learn from them, but write your own
code — that's the whole point.
```
