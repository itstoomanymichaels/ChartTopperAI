# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load the uploaded dataset
file_path = "/content/data.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns for model training
data_filtered = data.drop(columns=['id', 'track_name', 'album_name', 'artist', 'popularity'])

# Handling missing values
# Fill missing values in 'explicit' as 'False' and in 'mode' as 0
data_filtered['explicit'].fillna('False', inplace=True)
data_filtered['mode'].fillna(0, inplace=True)

# Encoding categorical variables
data_filtered['explicit'] = data_filtered['explicit'].map({'True': 1, 'False': 0})
data_filtered = pd.get_dummies(data_filtered, columns=['genre'], drop_first=True)

# Splitting the dataset into features and target variable
X = data_filtered.drop(columns=['target'])
y = data_filtered['target']

# Splitting the dataset into training and testing sets (80% train, 20% test)
# Control state is 42, if changed, it is advised to change in the other models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with the mean for numerical columns
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid_rf, n_iter=20, cv=3, n_jobs=-1, scoring='f1', random_state=42)
random_search_rf.fit(X_train_resampled, y_train_resampled)

print("Best parameters found for Random Forest: ", random_search_rf.best_params_)

# Train Random Forest with the best parameters
best_rf_model = random_search_rf.best_estimator_

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(best_rf_model, X_train_resampled, y_train_resampled, cv=5, scoring='f1')
print(f"Random Forest Cross-Validation F1 Scores: {cv_scores_rf}")
print(f"Random Forest Cross-Validation F1 Mean: {cv_scores_rf.mean()}")

best_rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf_train = best_rf_model.predict(X_train_scaled)
y_pred_rf = best_rf_model.predict(X_test_scaled)
y_pred_rf_prob = best_rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics for Random Forest
f1_rf_train = f1_score(y_train, y_pred_rf_train)
f1_rf = f1_score(y_test, y_pred_rf)
accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"Random Forest - F1 Score: {f1_rf}, AUC: {auc_rf}, Accuracy: {accuracy_rf}")

# Hyperparameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
random_search_xgb = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid_xgb, n_iter=20, cv=3, n_jobs=-1, scoring='f1', random_state=42)
random_search_xgb.fit(X_train_resampled, y_train_resampled)

print("Best parameters found for XGBoost: ", random_search_xgb.best_params_)

# Train XGBoost with the best parameters
best_xgb_model = random_search_xgb.best_estimator_

# Cross-validation for XGBoost
cv_scores_xgb = cross_val_score(best_xgb_model, X_train_resampled, y_train_resampled, cv=5, scoring='f1')
print(f"XGBoost Cross-Validation F1 Scores: {cv_scores_xgb}")
print(f"XGBoost Cross-Validation F1 Mean: {cv_scores_xgb.mean()}")

best_xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb_train = best_xgb_model.predict(X_train_scaled)
y_pred_xgb = best_xgb_model.predict(X_test_scaled)
y_pred_xgb_prob = best_xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics for XGBoost
f1_xgb_train = f1_score(y_train, y_pred_xgb_train)
f1_xgb = f1_score(y_test, y_pred_xgb)
accuracy_xgb_train = accuracy_score(y_train, y_pred_xgb_train)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_xgb)
print(f"XGBoost - F1 Score: {f1_xgb}, AUC: {auc_xgb}, Accuracy: {accuracy_xgb}")

# Logistic Regression with Randomized Search
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

lr_model = LogisticRegression(random_state=42, class_weight='balanced')
random_search_lr = RandomizedSearchCV(estimator=lr_model, param_distributions=param_grid_lr, n_iter=20, cv=3, n_jobs=-1, scoring='f1', random_state=42)
random_search_lr.fit(X_train_resampled, y_train_resampled)

print("Best parameters found for Logistic Regression: ", random_search_lr.best_params_)

# Train Logistic Regression with the best parameters
best_lr_model = random_search_lr.best_estimator_

# Cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(best_lr_model, X_train_resampled, y_train_resampled, cv=5, scoring='f1')
print(f"Logistic Regression Cross-Validation F1 Scores: {cv_scores_lr}")
print(f"Logistic Regression Cross-Validation F1 Mean: {cv_scores_lr.mean()}")

best_lr_model.fit(X_train_resampled, y_train_resampled)
y_pred_lr_train = best_lr_model.predict(X_train_scaled)
y_pred_lr = best_lr_model.predict(X_test_scaled)
y_pred_lr_prob = best_lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics for Logistic Regression
f1_lr_train = f1_score(y_train, y_pred_lr_train)
f1_lr = f1_score(y_test, y_pred_lr)
accuracy_lr_train = accuracy_score(y_train, y_pred_lr_train)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f"Logistic Regression - F1 Score: {f1_lr}, AUC: {auc_lr}, Accuracy: {accuracy_lr}")

# Summary of Results
print("\nSummary of Model Performance:")
print(f"Random Forest - F1 Score: {f1_rf}, AUC: {auc_rf}, Accuracy: {accuracy_rf}")
print(f"XGBoost - F1 Score: {f1_xgb}, AUC: {auc_xgb}, Accuracy: {accuracy_xgb}")
print(f"Logistic Regression - F1 Score: {f1_lr}, AUC: {auc_lr}, Accuracy: {accuracy_lr}")

# Plotting ROC Curves for all models
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_prob)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_prob)

plt.figure(figsize=(14, 6))

# ROC Curve
plt.subplot(1, 3, 1)
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(auc(fpr_rf, tpr_rf)))
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.2f})'.format(auc(fpr_xgb, tpr_xgb)))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = {:.2f})'.format(auc(fpr_lr, tpr_lr)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')

# F1 Score Comparison
plt.subplot(1, 3, 2)
models = ['Random Forest', 'XGBoost', 'Logistic Regression']
f1_scores_train = [f1_rf_train, f1_xgb_train, f1_lr_train]
f1_scores_test = [f1_rf, f1_xgb, f1_lr]
x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, f1_scores_train, width, label='Train', color='blue')
plt.bar(x + width/2, f1_scores_test, width, label='Test', color='red')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison')
plt.xticks(ticks=x, labels=models)
plt.legend()

# Accuracy Comparison
plt.subplot(1, 3, 3)
accuracies_train = [accuracy_rf_train, accuracy_xgb_train, accuracy_lr_train]
accuracies_test = [accuracy_rf, accuracy_xgb, accuracy_lr]
plt.bar(x - width/2, accuracies_train, width, label='Train', color='blue')
plt.bar(x + width/2, accuracies_test, width, label='Test', color='red')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.xticks(ticks=x, labels=models)
plt.legend()

plt.tight_layout()
plt.show()