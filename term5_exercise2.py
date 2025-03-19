# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Step 1: Load and preprocess the data
# Load the dataset
df = pd.read_csv('used_cars.csv')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (or handle them appropriately)
df.dropna(inplace=True)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Step 2: Data visualization
# Pairplot to visualize relationships between features
sns.pairplot(df)
plt.show()

# Heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Step 3: Split the data into training and testing sets
# Separate features (X) and target (y)
X = df.drop('price', axis=1)  # Replace 'price' with the actual target column name
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply different regression models
# 4.1 Simple Linear Regression (Univariate)
simple_lr = LinearRegression()
simple_lr.fit(X_train[['feature_name']], y_train)  # Replace 'feature_name' with an actual feature
y_pred_simple = simple_lr.predict(X_test[['feature_name']])

# 4.2 Multiple Linear Regression
multiple_lr = LinearRegression()
multiple_lr.fit(X_train, y_train)
y_pred_multiple = multiple_lr.predict(X_test)

# 4.3 Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
y_pred_poly = poly_lr.predict(X_test_poly)

# 4.4 Regularized Regression Models (Lasso, Ridge, ElasticNet)
# Lasso Regression
lasso = Lasso()
parameters_lasso = {'alpha': [0.01, 0.1, 1, 10]}
lasso_cv = GridSearchCV(lasso, parameters_lasso, cv=5)
lasso_cv.fit(X_train, y_train)
y_pred_lasso = lasso_cv.predict(X_test)

# Ridge Regression
ridge = Ridge()
parameters_ridge = {'alpha': [0.01, 0.1, 1, 10]}
ridge_cv = GridSearchCV(ridge, parameters_ridge, cv=5)
ridge_cv.fit(X_train, y_train)
y_pred_ridge = ridge_cv.predict(X_test)

# ElasticNet Regression
elastic = ElasticNet()
parameters_elastic = {'alpha': [0.01, 0.1, 1, 10], 'l1_ratio': [0.5, 0.7, 0.9]}
elastic_cv = GridSearchCV(elastic, parameters_elastic, cv=5)
elastic_cv.fit(X_train, y_train)
y_pred_elastic = elastic_cv.predict(X_test)

# Step 5: Evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'{model_name} - MSE: {mse}, R-squared: {r2}')

# Evaluate all models
evaluate_model(y_test, y_pred_simple, 'Simple Linear Regression')
evaluate_model(y_test, y_pred_multiple, 'Multiple Linear Regression')
evaluate_model(y_test, y_pred_poly, 'Polynomial Regression')
evaluate_model(y_test, y_pred_lasso, 'Lasso Regression')
evaluate_model(y_test, y_pred_ridge, 'Ridge Regression')
evaluate_model(y_test, y_pred_elastic, 'ElasticNet Regression')

# Step 6: Compare results using plots
# Compare MSE values
models = ['Simple LR', 'Multiple LR', 'Polynomial LR', 'Lasso', 'Ridge', 'ElasticNet']
mse_scores = [
    mean_squared_error(y_test, y_pred_simple),
    mean_squared_error(y_test, y_pred_multiple),
    mean_squared_error(y_test, y_pred_poly),
    mean_squared_error(y_test, y_pred_lasso),
    mean_squared_error(y_test, y_pred_ridge),
    mean_squared_error(y_test, y_pred_elastic)
]

plt.figure(figsize=(10, 6))
plt.bar(models, mse_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of Regression Models')
plt.show()

# Optional: Cross-Validation for better evaluation
cv_scores = cross_val_score(multiple_lr, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validation MSE (Multiple LR): {-cv_scores.mean()}')