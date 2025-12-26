📘 Multiple Linear Regression – Practical Recall Notes
======================================================

1️⃣ What is Multiple Linear Regression?
---------------------------------------

**Definition**Multiple Linear Regression is used when **more than one independent feature** is used to predict a single continuous output.

y^=β0+β1x1+β2x2+⋯+βkxk\\hat{y} = \\beta\_0 + \\beta\_1 x\_1 + \\beta\_2 x\_2 + \\dots + \\beta\_k x\_ky^​=β0​+β1​x1​+β2​x2​+⋯+βk​xk​

**When to use**

*   Output is continuous
    
*   Relationship is approximately linear
    
*   More than one influencing feature exists
    

2️⃣ Dataset Understanding (What & Why)
--------------------------------------

**Dataset:** economic\_index.csv

**Features**

*   interest\_rate → independent
    
*   unemployment\_rate → independent
    
*   index\_price → dependent (target)
    

**Why drop year, month, Unnamed: 0**

*   They do not influence the target directly
    
*   Keeping them adds noise
    
*   May cause misleading correlations
    

3️⃣ Why Data Visualization First
--------------------------------

### Pairplot & Correlation

*   **Why**: To verify linear relationship and detect patterns
    
*   **What it tells**:
    
    *   Strength of relationship
        
    *   Direction (positive/negative)
        
    *   Presence of multicollinearity
        

High correlation with target → good predictorHigh correlation among predictors → potential multicollinearity

4️⃣ Feature Selection (Why X is 2D)
-----------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   X = df[['interest_rate', 'unemployment_rate']]  y = df['index_price']   `

**Why X must be 2D**

*   sklearn expects shape (n\_samples, n\_features)
    
*   Consistent interface for single or multiple features
    

**Why y can be 1D**

*   Target is a single value per sample
    

5️⃣ Train–Test Split (Why & When)
---------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   train_test_split(test_size=0.25, random_state=42)   `

**Why**

*   To test generalization
    
*   Prevent overfitting illusion
    

**random\_state = 42**

*   Ensures reproducibility
    
*   Same split every run
    

6️⃣ Why Standard Scaling is Important
-------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   StandardScaler()   `

**Why**

*   Gradient descent converges faster
    
*   Features on different scales do not dominate
    
*   Improves numerical stability
    

**Golden rule**

*   fit\_transform() → training data
    
*   transform() → test & future data(prevents data leakage)
    

7️⃣ Model Training (What Happens Internally)
--------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   LinearRegression().fit(X_train, y_train)   `

*   sklearn uses **Ordinary Least Squares (OLS)**
    
*   Solves equation directly (no iterations)
    

Model learns coefficients:

β=(XTX)−1XTy\\beta = (X^T X)^{-1} X^T yβ=(XTX)−1XTy

8️⃣ Meaning of Coefficients
---------------------------

*   **Coefficient (βi\\beta\_iβi​)** → change in target per unit change in feature
    
*   **Intercept (β0\\beta\_0β0​)** → predicted value when all features = 0
    

Interpretation must be done **after scaling context is understood**.

9️⃣ Why Cross-Validation is Used
--------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   cross_val_score(scoring='neg_mean_squared_error')   `

**Why**

*   Single train-test split may be misleading
    
*   Cross-validation gives average performance
    
*   Reduces variance in evaluation
    

**Negative MSE**

*   sklearn maximizes scores
    
*   Negative sign allows MSE to fit maximization framework
    

🔟 Evaluation Metrics (When to Use What)
----------------------------------------

MetricWhyMAEEasy interpretationMSEPenalizes large errorsRMSESame unit as targetR²Variance explainedAdjusted R²Penalizes extra features

**Adjusted R² is preferred in multiple regression**

1️⃣1️⃣ Residual Analysis (Assumptions Check)
--------------------------------------------

### What to check

*   Residuals should be:
    
    *   Normally distributed
        
    *   Mean ≈ 0
        
    *   No clear pattern vs predictions
        

**Why**

*   Validates linear regression assumptions
    
*   Detects heteroscedasticity & non-linearity
    

1️⃣2️⃣ OLS with Statsmodels (Why Compare)
-----------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   statsmodels.OLS()   `

**Why**

*   Detailed statistical summary
    
*   Confirms sklearn results
    
*   Shows:
    
    *   p-values
        
    *   F-statistic
        
    *   confidence intervals
        

If coefficients ≈ sklearn → model is correct.

1️⃣3️⃣ Key Practical Rules (Must Remember)
------------------------------------------

*   Always scale features when using GD-based methods
    
*   Never fit scaler on test data
    
*   X → always 2D
    
*   y → always 1D
    
*   Validate assumptions using residuals
    
*   Adjusted R² > R² for model comparison