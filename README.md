📉 Telco Customer Churn Prediction & Deployment

An end-to-end machine learning project that predicts which telecom customers are likely to cancel their subscription, so a company can target retention offers. It includes a full modeling workflow and a Streamlit web app for real-time predictions.

🔗 Live demo: paste your Streamlit link here

App screenshot

Problem

Acquiring a new customer costs far more than keeping an existing one. About 26% of customers in this dataset churn, so the goal is to flag at-risk customers early. Because the classes are imbalanced, accuracy alone is misleading, so the project focuses on Recall, F1 and ROC-AUC for the churn class.

Dataset

Telco Customer Churn (Kaggle, IBM sample data): 7,043 customers, 19 features covering demographics, services, contract and billing. The data file is not included in this repo; download it from Kaggle to run the notebook.

Results (held-out test set, 20%)
Metric (churn class)	Score
Recall	0.805
Precision	0.506
F1-score	0.621
Accuracy	0.739
ROC-AUC	0.837

Selected model: XGBoost, tuned with GridSearchCV (learning_rate=0.05, max_depth=5, n_estimators=100, scale_pos_weight=3). The class weighting trades some precision for a much higher recall on churners.

Confusion matrix and ROC curve

Key findings
Contract type is the strongest driver: month-to-month customers churn at roughly 43%, versus about 11% on one-year and 3% on two-year contracts.
New customers leave the most: churn is concentrated in the first months of tenure, where more than half of the newest customers leave.
Internet service and payment method matter: fiber-optic customers churn at 41.9% (DSL 19.0%, no internet 7.4%), and electronic-check payers at 45.3% (15-19% for other methods).
Churners pay more: their median monthly charge is higher than that of customers who stay.
Top model features (permutation importance): Contract_Month-to-month, tenure, TotalCharges, OnlineSecurity_No, InternetService_Fiber optic.
Threshold trade-off: at the default 0.5 the model catches 80% of churners at 51% precision. Lowering the threshold to 0.3 raises recall to 90% (missing only 36 of 374 churners) but flags 804 of 1,407 customers; raising it to 0.7 gives 62% precision but misses 152 churners. The right value depends on the cost of a retention offer versus a lost customer.
Workflow
Cleaning: converted TotalCharges to numeric, removed 11 rows with blank values, encoded the target.
EDA: churn by contract, tenure, monthly charges, internet service and payment method.
Feature engineering: n_services (number of add-on services), tenure_group (customer lifetime bucket), avg_charge (average paid per month of tenure).
Model comparison: 7 models (Logistic Regression, SVM, KNN, Decision Tree, Random Forest, AdaBoost, XGBoost) compared with 5-fold stratified cross-validation on the training set only.
Tuning: GridSearchCV on the top 3 models; the winner is chosen by cross-validated F1, never by the test set.
Evaluation: test set used once. Precision, Recall, F1, ROC-AUC, confusion matrix, Precision-Recall curve and a threshold analysis.
Interpretation: permutation importance to see what drives churn.
Deployment: Streamlit app that applies the same preprocessing as training and returns the churn probability.
Design choices
Cross-validation for model selection, so the test score is an honest estimate.
Scaler fitted on training folds only (inside a Pipeline during CV/tuning) to avoid data leakage.
Recall / F1 / ROC-AUC instead of accuracy because of class imbalance.
Threshold analysis: the notebook shows how recall and precision change with the decision threshold, so the business can pick the trade-off that matches the cost of losing a customer.
Project structure
├── churn_prediction.ipynb    # analysis, training and evaluation
├── app.py                    # Streamlit app
├── xgb_model.json            # trained XGBoost model
├── scaler.pkl                # fitted StandardScaler
├── feature_columns.json      # training column order used by the app
├── requirements.txt
├── .streamlit/config.toml    # app theme
└── README.md
Run locally
bash
git clone https://github.com/Habiba-elgamal158/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt
python -m streamlit run app.py

To retrain the model, download the dataset from Kaggle, place Telco-Customer-Churn.csv next to the notebook and run churn_prediction.ipynb.

Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit
