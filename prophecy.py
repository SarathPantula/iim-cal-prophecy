import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load your dataset
# Replace with your actual dataset file
df = pd.read_csv('real_estate_data.csv')

# Feature Engineering and Selection
# Assume these are the columns in your dataset. Adjust as per your dataset
numerical_features = ['size', 'age', 'num_bedrooms', 'num_bathrooms']
categorical_features = ['neighborhood', 'property_type']
target_feature = 'price'

# Preprocessing for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models
linear_model = LinearRegression()
lasso_model = Lasso()
ridge_model = Ridge()

# Create a preprocessing and modeling pipeline
linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', linear_model)])

lasso_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', lasso_model)])

ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', ridge_model)])

# Splitting the dataset into training and testing sets
X = df[numerical_features + categorical_features]
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training and predictions
linear_pipeline.fit(X_train, y_train)
lasso_pipeline.fit(X_train, y_train)
ridge_pipeline.fit(X_train, y_train)

linear_predictions = linear_pipeline.predict(X_test)
lasso_predictions = lasso_pipeline.predict(X_test)
ridge_predictions = ridge_pipeline.predict(X_test)

# Model evaluation
linear_mse = mean_squared_error(y_test, linear_predictions)
lasso_mse = mean_squared_error(y_test, lasso_predictions)
ridge_mse = mean_squared_error(y_test, ridge_predictions)

linear_r2 = r2_score(y_test, linear_predictions)
lasso_r2 = r2_score(y_test, lasso_predictions)
ridge_r2 = r2_score(y_test, ridge_predictions)

print("Linear Regression - MSE:", linear_mse, "R2:", linear_r2)
print("Lasso Regression - MSE:", lasso_mse, "R2:", lasso_r2)
print("Ridge Regression - MSE:", ridge_mse, "R2:", ridge_r2)

# Hyperparameter tuning for Lasso and Ridge using GridSearchCV
parameters = {'model__alpha': [0.01, 0.1, 1, 10, 100]}

lasso_grid_search = GridSearchCV(lasso_pipeline, parameters, cv=5)
ridge_grid_search = GridSearchCV(ridge_pipeline, parameters, cv=5)

lasso_grid_search.fit(X_train, y_train)
ridge_grid_search.fit(X_train, y_train)

print("Best parameters for Lasso:", lasso_grid_search.best_params_)
print("Best parameters for Ridge:", ridge_grid_search.best_params_)

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(y_test, linear_predictions, alpha=0.5, label='Linear', color='blue')
plt.scatter(y_test, lasso_predictions, alpha=0.5, label='Lasso', color='green')
plt.scatter(y_test, ridge_predictions, alpha=0.5, label='Ridge', color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices: Linear, Lasso, Ridge')
plt.legend()
plt.show()
