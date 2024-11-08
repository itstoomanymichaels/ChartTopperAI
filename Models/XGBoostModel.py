# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
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

# Initialize and train the XGBoost Classifier model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test_scaled)

# Calculate the RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE for XGBoost Classifier: {rmse}")
