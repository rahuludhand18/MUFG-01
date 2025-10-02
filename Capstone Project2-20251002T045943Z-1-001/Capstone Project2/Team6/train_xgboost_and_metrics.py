import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from xgboost import XGBClassifier

# Load cleaned data
df = pd.read_csv('data/heart_clean.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter search
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=10,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
print("Best params:", search.best_params_)

# Evaluate best model
model = search.best_estimator_
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)
print("True Healthy (TN):", tn)
print("False Sick (FP):", fp)
print("False Healthy (FN):", fn)
print("True Sick (TP):", tp)

# Save model
joblib.dump(model, 'models/best_model.pkl')
print("Model saved to models/best_model.pkl")

# Save metrics table as CSV for Streamlit
metrics = {
    "Model": ["XGBoost"],
    "Accuracy": [f"{accuracy*100:.1f}%"],
    "Precision": [f"{precision*100:.1f}%"],
    "Recall": [f"{recall*100:.1f}%"],
    "F1 Score": [f"{f1*100:.1f}%"],
    "ROC-AUC": [f"{roc_auc*100:.1f}%"],
    "True Healthy": [tn],
    "False Sick": [fp],
    "False Healthy": [fn],
    "True Sick": [tp]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('models/model_metrics.csv', index=False)
print("Metrics saved to models/model_metrics.csv")