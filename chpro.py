import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Data Cleaning and Preprocessing
# Handling missing values
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Encoding categorical variables
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Ensure test data has the same features as train data
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0  # Add missing columns with default values

test_data = test_data[train_data.columns.drop("Item_Outlet_Sales", errors='ignore')]

# Feature Selection
X = train_data.drop('Item_Outlet_Sales', axis=1)
y = train_data['Item_Outlet_Sales']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Model Training and Evaluation

# 1. Ridge Regression (Regularized Linear Regression)
lr_model = Ridge(alpha=1.0)
lr_model.fit(X_train, y_train)
lr_train_pred = lr_model.predict(X_train)
lr_val_pred = lr_model.predict(X_val)
lr_train_rmse = root_mean_squared_error(y_train, lr_train_pred)
lr_val_rmse = root_mean_squared_error(y_val, lr_val_pred)
print(f"Ridge Regression - Train RMSE: {lr_train_rmse:.2f}, Validation RMSE: {lr_val_rmse:.2f}")

# 2. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
gb_search = HalvingGridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, factor=2, min_resources=20)
gb_search.fit(X_train, y_train)
gb_best_model = gb_search.best_estimator_

gb_train_pred = gb_best_model.predict(X_train)
gb_val_pred = gb_best_model.predict(X_val)
gb_train_rmse = root_mean_squared_error(y_train, gb_train_pred)
gb_val_rmse = root_mean_squared_error(y_val, gb_val_pred)
print(f"Gradient Boosting - Train RMSE: {gb_train_rmse:.2f}, Validation RMSE: {gb_val_rmse:.2f}")

# Make Predictions on Test Data
predictions = gb_best_model.predict(test_data)

# Save Predictions
output_df = pd.DataFrame({
    'Item_Identifier': test_data.index,  # Ensure 'Item_Identifier' exists
    'Item_Outlet_Sales': predictions
})
output_df.to_csv("predictions.csv", index=False)