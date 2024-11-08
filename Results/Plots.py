# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load the uploaded dataset
file_path = "../Merger/data.csv"
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with the mean for numerical columns
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize models
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train models
xgb_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
models = {'XGBoost Classifier': xgb_model, 'Random Forest Classifier': rf_model, 'Logistic Regression': lr_model}

# Initialize lists to store metrics
accuracy_scores = []
f1_scores = []
roc_auc_scores = []

# Plotting ROC curves
plt.figure(figsize=(10, 8))

for model_name, model in models.items():
    y_pred = model.predict(X_test_imputed)
    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Append metrics to lists
    accuracy_scores.append((model_name, accuracy))
    f1_scores.append((model_name, f1))
    roc_auc_scores.append((model_name, roc_auc))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot settings for ROC curves
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Models')
plt.legend()
plt.show()

# Plot bar charts for Accuracy, F1 Score, and ROC AUC for each model
# Accuracy Chart
plt.figure(figsize=(10, 6))
model_names = [name for name, _ in accuracy_scores]
accuracy_values = [score for _, score in accuracy_scores]
plt.bar(model_names, accuracy_values, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Model Comparison - Accuracy')
plt.show()

# F1 Score Chart
plt.figure(figsize=(10, 6))
f1_values = [score for _, score in f1_scores]
plt.bar(model_names, f1_values, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Model Comparison - F1 Score')
plt.show()

# ROC AUC Chart
plt.figure(figsize=(10, 6))
roc_auc_values = [score for _, score in roc_auc_scores]
plt.bar(model_names, roc_auc_values, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Model Comparison - ROC AUC')
plt.show()