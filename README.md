# Custom ML Classifiers From Scratch: Vectorized k-NN & SVM


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohdAli21/Building-SVM-and-k-NN-Classifiers-from-Scratch/blob/main/Custom_KNN_and_SVM_From_Scratch.ipynb) 

A mathematically rigorous, library-free implementation of **k-Nearest Neighbors (k-NN)** and **Support Vector Machines (SVM)** built entirely from scratch in Python. By intentionally bypassing high-level packages like `scikit-learn` for model architecture, this project serves to expose and document the underlying optimization loops, high-dimensional projections, and geometric vectors that govern these algorithms.

Both systems leverage **NumPy matrix broadcasting vectorization** to optimize processing speeds over dense feature grids, demonstrating production-ready performance without external optimization dependencies.

---

## 🚀 Core Architectural Highlights

### 1. k-Nearest Neighbors (k-NN) Engine
* **Vectorized Distance Diagnostics:** Eliminated traditional, sluggish nested Python loops by computing multi-dimensional spatial distances simultaneously across matrix axes for both **Euclidean ($L_2$)** and **Manhattan ($L_1$)** norms.
* **Hyperparameter Tuning via the Elbow Method:** Engineered a manual validation pipeline to analyze hyperparameter variance. By graphing classification error drops across odd $k$-values, the system dynamically identified **$k=17$** as the optimal boundary threshold for non-linear data, neutralizing local noise spikes.

### 2. Support Vector Machine (SVM) Engine
* **The Manual Kernel Trick:** Implemented structural kernel matrices to project non-linear geometries into high-dimensional separable planes using custom functions:
  * **Linear Kernel:** $K(x, y) = x^T y$
  * **Polynomial Kernel:** $K(x, y) = (x^T y + c)^d$
  * **Radial Basis Function (RBF) Kernel:** Employs manual algebraic expansion ($\|x_1 - x_2\|^2 = \|x_1\|^2 + \|x_2\|^2 - 2x_1x_2^T$) to perform instantaneous Gaussian spatial clustering across dense mesh grids.
* **Stochastic Gradient Descent (SGD) Solver:** Implemented an explicit soft-margin training loop optimizing a dual-condition **Hinge Loss** function, dynamically shifting the maximum-margin boundary weights ($\alpha$) and bias ($b$) vector spaces over thousands of epochs.

---

## 📊 Performance Benchmarks
Evaluated on complex, interlocking non-linear synthetic geometries (the Moons dataset) under a 30% unmapped data split:
* **Custom k-NN Model Accuracy:** `94.44%` (Stabilized via Elbow Method analysis)
* **Custom RBF SVM Model Accuracy:** `94.44%` (Winding, fluid non-linear separation path)

---

## 🛠️ Project Setup & Installation

To run this notebook or replicate the standalone modules locally, clone the repository and configure the minimal mathematical environment:

```bash
# Clone the repository
git clone https://github.com/MohdAli21/Building-SVM-and-k-NN-Classifiers-from-Scratch
cd Building-SVM-and-k-NN-Classifiers-from-Scratch

# Install minimal core dependencies
pip install numpy matplotlib sklearn
