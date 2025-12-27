14️⃣ Grid Search Hyperparameter Tuning – Reminder Notes
=======================================================

1️⃣ Why Hyperparameter Tuning?
------------------------------

*   Default parameters ≠ best parameters
    
*   Different **random\_state → different split → different accuracy**
    
*   Goal: **find the most stable + best-performing model**
    

2️⃣ What Are Hyperparameters?
-----------------------------

*   Parameters **not learned** by model
    
*   Set **before training**
    
*   Control **bias–variance tradeoff**
    

Examples in Logistic Regression:

*   penalty
    
*   C
    
*   solver
    
*   class\_weight
    

3️⃣ Meaning of Important Logistic Regression Params
---------------------------------------------------

ParameterReminderpenaltyType of regularization (l1, l2, elasticnet)CInverse of regularization strength (↓C = ↑regularization)solverOptimization algorithmclass\_weightHandle imbalance

4️⃣ Why GridSearchCV?
---------------------

> Tries **ALL possible combinations** of parametersSelects the **best one using cross-validation**

✔ Exhaustive❌ Slower for large grids

5️⃣ Parameter Grid (Most Important Step)
----------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   params = {      'penalty': ['l1', 'l2'],      'C': [100, 10, 1, 0.1, 0.01],      'solver': ['liblinear', 'saga']  }   `

📌 **Keys must EXACTLY match model parameter names**

6️⃣ Why Not Any Solver With Any Penalty?
----------------------------------------

Some combinations are **invalid**

*   liblinear → supports l1, l2
    
*   saga → supports l1, l2, elasticnet
    

📌 Always check sklearn docs

7️⃣ Cross-Validation (Why Needed?)
----------------------------------

*   Single train-test split is unreliable
    
*   CV gives **robust performance estimate**
    

8️⃣ Stratified K-Fold (Very Important)
--------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from sklearn.model_selection import StratifiedKFold  cv = StratifiedKFold(n_splits=5)   `

Why stratified?

*   Keeps **class ratio same** in every fold
    
*   Prevents biased evaluation
    

9️⃣ Creating GridSearchCV Object
--------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from sklearn.model_selection import GridSearchCV  grid = GridSearchCV(      estimator=model,      param_grid=params,      scoring='accuracy',      cv=cv,      n_jobs=-1  )   `

### What Each Argument Means

ArgumentMeaningestimatorModel to tuneparam\_gridParameters to tryscoringMetric to optimizecvCross-validation strategyn\_jobs=-1Use all CPU cores

🔟 Training Grid Search
-----------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   grid.fit(X_train, y_train)   `

What happens internally:

*   Try every param combo
    
*   Apply CV for each
    
*   Store scores
    
*   Pick best combo
    

1️⃣1️⃣ Getting Best Parameters & Score
--------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   grid.best_params_  grid.best_score_   `

📌 best\_score\_ → CV score (not test score)

1️⃣2️⃣ Making Predictions with Best Model
-----------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   y_pred = grid.predict(X_test)   `

📌 Uses **best\_estimator\_ internally**

1️⃣3️⃣ Why GridSearchCV Improves Performance?
---------------------------------------------

*   Reduces overfitting
    
*   Finds optimal regularization
    
*   Balances bias & variance
    

1️⃣4️⃣ When NOT to Use GridSearchCV?
------------------------------------

*   Huge datasets
    
*   Very large parameter space
    

👉 Use **RandomizedSearchCV** instead (next topic)

🧠 One-Line Mental Model
------------------------

> **GridSearchCV = brute-force + cross-validated parameter optimization**