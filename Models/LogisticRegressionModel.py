# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
# Control state is 42, if changed, it is advised to change in the other models
# as well to keep consistency
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with the mean for numerical columns
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],        # Regularization types
    'solver': ['liblinear', 'saga'] # Solvers compatible with l1/l2 penalties
}
grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_lr_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)


# Initialize and train the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = best_lr_model.predict(X_test_scaled)
y_pred_proba = best_lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model performance
accuracy = best_lr_model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Classification report for precision, recall, F1-score
print(classification_report(y_test, y_pred))

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# Calculate the RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE for Linear Regression: {rmse}")

# Cross-validation to check robustness
cv_scores = cross_val_score(best_lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-validated Accuracy: {np.mean(cv_scores):.4f}")