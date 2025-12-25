📘 Simple Linear Regression – Practical Recall Notes

1️⃣ Why X must be 2D and y can be 1D

In scikit-learn, X = features has shape (n_samples, n_features) (2D), while y = target has shape (n_samples, ) (1D).

Why? ML models are designed for multiple features. Even if there is one feature, sklearn expects the same interface.

X = df[['Weight']]   # ✅ 2D DataFrame
y = df['Height']     # ✅ 1D Series


❌ Using df['Weight'] as X breaks consistency.
Mental rule: One sample = one row, one feature = one column → always 2D.

2️⃣ Why random_state = 42

train_test_split(X, y, test_size=0.25, random_state=42)

Why? The train-test split is random. Without random_state, every run gives a different split, causing different models and confusing results. random_state ensures reproducibility.
📌 42 is just a convention. Any number works.

3️⃣ Why Train–Test Split is mandatory

Goal: Test model on unseen data. If you train and test on the same data, the model looks perfect but is useless in the real world (Overfitting Illusion).
Rule: Train → Learn patterns. Test → Check generalization.

4️⃣ Why Standardization is used

Formula (z-score): 

$$z = \frac{x - \mu}{\sigma}$$


Why? It helps gradient descent converge faster, prevents large-scale features from dominating, and improves numerical stability.
⚠️ Note: For OLS (closed-form), it's a good habit but not mandatory. For Gradient Descent, it is very important.

5️⃣ Why fit_transform on train but only transform on test

scaler.fit_transform(X_train)  # ✅ Learns mean/std & applies
scaler.transform(X_test)       # ✅ Applies learned mean/std


Why? If you fit on test data, test data leaks into training (Data Leakage), making results invalid.
Golden rule: Never learn anything from test data.

6️⃣ Why sklearn LinearRegression works without gradient descent

LinearRegression() uses OLS (Normal Equation) internally: 

$$\theta = (X^T X)^{-1} X^T y$$


That means: No iterations, no learning rate, and a direct mathematical solution. This is why scaling isn't strictly required for this specific algorithm and why it's fast for small datasets.

7️⃣ Meaning of Coefficient and Intercept

regressor.coef_ (slope $\beta_1$) and regressor.intercept_ (intercept $\beta_0$).
Equation: 

$$\hat{y} = \beta_0 + \beta_1 x$$


Coefficient: Change in $y$ for 1 unit change in $x$.
Intercept: Predicted $y$ when $x = 0$ (mathematically needed, even if physically meaningless).

8️⃣ Why Visualization is important

Scatter plot checks if linear regression is valid. Line plot confirms model fit visually.
It helps catch non-linearity, outliers, and bad assumptions. 📌 ML is visual + mathematical.

9️⃣ Error Metrics – When to use what

MAE: Average absolute mistake (easy to interpret).

MSE: Penalizes large errors more.

RMSE: Same unit as target (preferred).
📉 Lower = Better.

🔟 R² vs Adjusted R²

R²: % of variance explained. Always increases with more features. Use for basic understanding.

Adjusted R²: Penalizes unnecessary features. Use for model selection and comparison.

1️⃣1️⃣ Why predict new data must be standardized

If the model was trained on standardized data, you must transform new inputs too:

new_x = scaler.transform([[72]])
regressor.predict(new_x)

❌ Using raw input gives wrong output. Rule: Same preprocessing for train, test, and future data.

1️⃣2️⃣ sklearn vs OLS (statsmodels)
sklearn: Fast, production-ready, less stats.

statsmodels: Detailed statistics, research-focused, slower on large data. Both give the same coefficients → confirms correctness.