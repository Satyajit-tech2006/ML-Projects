📘 Simple Linear Regression – Practical Recall Notes
1️⃣ Why X must be 2D and y can be 1D

In scikit-learn:

X = features → shape (n_samples, n_features) → 2D

y = target → shape (n_samples, ) → 1D

Why?

ML models are designed for multiple features

Even if there is one feature, sklearn expects the same interface

X = df[['Weight']]   # ✅ 2D DataFrame
y = df['Height']     # ✅ 1D Series


❌ Using df['Weight'] as X breaks consistency.

Mental rule:

One sample = one row, one feature = one column → always 2D.

2️⃣ Why random_state = 42
train_test_split(X, y, test_size=0.25, random_state=42)

Why?

Train-test split is random

Without random_state, every run gives different split

That causes:

Different model

Different metrics

Confusing results

random_state = fixed seed → reproducibility

📌 42 is just a convention. Any number works.

3️⃣ Why Train–Test Split is mandatory

Goal: Test model on unseen data

If you train and test on same data:

Model looks perfect

Actually useless in real world

This is called overfitting illusion

Rule:

Train → Learn patterns
Test → Check generalization

4️⃣ Why Standardization is used
Formula (z-score)
𝑧
=
𝑥
−
𝜇
𝜎
z=
σ
x−μ
	​

Why standardize?

Gradient descent converges faster

Prevents features with large scale from dominating

Improves numerical stability

⚠️ For OLS (closed-form):

Standardization is not mandatory

But still a good habit

⚠️ For Gradient Descent:

Standardization is very important

5️⃣ Why fit_transform on train but only transform on test
scaler.fit_transform(X_train)  # ✅
scaler.transform(X_test)       # ✅

Why?

fit → learns mean & std

transform → applies them

❌ If you fit on test data:

Test data leaks into training

Results become invalid

This is called data leakage

Golden rule:

Never learn anything from test data.

6️⃣ Why sklearn LinearRegression works without gradient descent

LinearRegression() uses OLS (Normal Equation) internally.

That means:

No iterations

No learning rate

Direct mathematical solution

Equation:

𝜃
=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑦
θ=(X
T
X)
−1
X
T
y

That’s why:

Scaling not strictly required

Very fast for small datasets

7️⃣ Meaning of Coefficient and Intercept
regressor.coef_       # slope (β₁)
regressor.intercept_  # intercept (β₀)


Equation:

𝑦
^
=
𝛽
0
+
𝛽
1
𝑥
y
^
	​

=β
0
	​

+β
1
	​

x

Coefficient → change in y for 1 unit change in x

Intercept → predicted y when x = 0
(may not be meaningful physically, but mathematically needed)

8️⃣ Why Visualization is important

Scatter plot → checks if linear regression is valid

Line plot → confirms model fit visually

Helps catch:

Non-linearity

Outliers

Bad assumptions

📌 ML is visual + mathematical

9️⃣ Error Metrics – When to use what

MAE → average absolute mistake (easy to interpret)

MSE → penalizes large errors more

RMSE → same unit as target (preferred)

Lower = better.

🔟 R² vs Adjusted R²
R²

% of variance explained

Always increases with more features

Adjusted R²

Penalizes unnecessary features

Better for model comparison

Use:

R² → basic understanding

Adjusted R² → model selection

1️⃣1️⃣ Why predict new data must be standardized

If model trained on standardized data:

new_x = scaler.transform([[72]])
regressor.predict(new_x)


❌ Using raw input gives wrong output

Rule:

Same preprocessing for train, test, and future data.

1️⃣2️⃣ sklearn vs OLS (statsmodels)
sklearn	statsmodels
Fast	Detailed statistics
Production-ready	Research/analysis
Less stats	Full statistical summary

Both give same coefficients → confirms correctness.