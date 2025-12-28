# -----------------------------------------
# Logistic Regression – Perfect Visual Demo
# -----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1️⃣ Create a binary classification dataset (2 features for visualization)
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# 2️⃣ Visualize the raw data
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Raw Classification Data")
plt.show()

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4️⃣ Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5️⃣ Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# 6️⃣ Predict probabilities over grid
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# 7️⃣ Plot decision boundary (best-fit logistic surface)
plt.figure(figsize=(7, 6))

# Probability contour
plt.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.6)

# Training points
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="bwr",
    edgecolors="k",
    label="Training Data"
)

# Decision boundary (P = 0.5)
plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression – Best Fit Decision Boundary")
plt.legend()
plt.show()
