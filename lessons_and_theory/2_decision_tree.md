# Lesson 2: Decision Trees and Random Forests for House Price Prediction

## Introduction

Imagine you're a data analyst tasked with predicting house prices in London based on various features like the number of bedrooms, area, location, etc. You need a model that not only provides accurate predictions but also offers interpretability to understand the impact of each feature on the price. This is where **Decision Trees** and **Random Forests** come into play.

A **Decision Tree** is a supervised machine learning algorithm used for both classification and regression tasks. It splits the data into subsets based on the value of input features, creating a tree-like structure of decisions that lead to a final prediction.

A **Random Forest** is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees for regression tasks.

In this lesson, we'll explore how to build decision tree and random forest models using Python to predict London house prices. We'll use the `London.csv` dataset, which contains information about various properties in London.

---

## Table of Contents

1. [Understanding Decision Trees and Random Forests](#understanding-decision-trees-and-random-forests)
2. [Loading and Exploring the Data](#loading-and-exploring-the-data)
3. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Preparing Data for Modeling](#preparing-data-for-modeling)
6. [Training and Evaluating the Decision Tree Model](#training-and-evaluating-the-decision-tree-model)
7. [Training and Evaluating the Random Forest Model](#training-and-evaluating-the-random-forest-model)
8. [Model Comparison and Interpretation](#model-comparison-and-interpretation)
9. [Advanced Techniques](#advanced-techniques)
10. [Conclusion](#conclusion)
11. [Further Reading](#further-reading)

---

## Understanding Decision Trees and Random Forests

### Decision Trees

Decision Trees work by recursively splitting the data into subsets based on the most significant attribute at each node. The goal is to create subsets that are as pure as possible in terms of the target variable.

Key Concepts:
- **Root Node**: The top node representing the entire dataset.
- **Internal Nodes**: Nodes that represent tests on features.
- **Leaf Nodes**: Terminal nodes that provide the final prediction.
- **Branches**: Edges that connect nodes, representing the outcome of a test.

Splitting Criteria:
- **Mean Squared Error (MSE)**: Used for regression trees to measure the variance within the subsets.
- **Gini Impurity** and **Entropy**: Used in classification trees to measure the impurity of a node.

### Random Forests

Random Forests are an ensemble of decision trees, where each tree is trained on a random subset of the data and features. This approach helps to reduce overfitting and improve generalization.

Key Concepts:
- **Bagging**: Bootstrap Aggregating, the process of randomly sampling subsets of the data with replacement.
- **Feature Randomness**: At each split, only a random subset of features is considered.
- **Ensemble Prediction**: The final prediction is an average of all individual tree predictions.

---

## Loading and Exploring the Data

First, let's load the dataset and take a look at its structure.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('London.csv')

# Display basic information about the dataset
print(df.info())
print("\nSample data:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())
```

[Include output from the above code here]

---

## Data Cleaning and Preprocessing

Based on the initial exploration, we'll perform necessary cleaning steps.

```python
# Handle missing values
df['Location'].fillna('Unknown', inplace=True)

# Convert price to numeric, removing any non-numeric characters
df['Price'] = pd.to_numeric(df['Price'].replace({'\£': '', ',': ''}, regex=True), errors='coerce')

# Convert area to numeric
df['Area in sq ft'] = pd.to_numeric(df['Area in sq ft'], errors='coerce')

# Drop rows with NaN values in important columns
df.dropna(subset=['Price', 'Area in sq ft', 'No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions'], inplace=True)

# Reset index after dropping rows
df.reset_index(drop=True, inplace=True)

print(f"After cleaning: {len(df)} rows remaining")
```

[Include output from the above code here]

---

## Exploratory Data Analysis (EDA)

Let's visualize some key aspects of our data.

```python
# Distribution of house prices
plt.figure(figsize=(10,6))
sns.histplot(df['Price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.show()

# Log-transformed price distribution
plt.figure(figsize=(10,6))
sns.histplot(np.log(df['Price']), kde=True)
plt.title('Distribution of Log-Transformed House Prices')
plt.xlabel('Log(Price)')
plt.show()

# Correlation heatmap
numeric_cols = ['Price', 'Area in sq ft', 'No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions']
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# Price vs Area scatter plot
plt.figure(figsize=(10,6))
sns.scatterplot(x='Area in sq ft', y='Price', data=df)
plt.title('Price vs Area')
plt.show()

# Average price by house type
avg_price_by_type = df.groupby('House Type')['Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,6))
avg_price_by_type.plot(kind='bar')
plt.title('Average Price by House Type')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()

# Top 10 locations by average price
top_locations = df.groupby('Location')['Price'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
top_locations.plot(kind='bar')
plt.title('Top 10 Locations by Average Price')
plt.xlabel('Location')
plt.ylabel('Average Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

[Include the following plots here:
1. Distribution of House Prices
2. Distribution of Log-Transformed House Prices
3. Correlation Heatmap of Numeric Features
4. Price vs Area Scatter Plot
5. Average Price by House Type
6. Top 10 Locations by Average Price]

---

## Preparing Data for Modeling

Now, let's prepare our data for modeling.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing steps
numeric_features = ['Area in sq ft', 'No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions']
categorical_features = ['House Type', 'Location', 'City/County']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# Fit and transform the data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

print("Training set shape:", X_train_processed.shape)
print("Testing set shape:", X_test_processed.shape)
```

[Include output from the above code here]

---

## Training and Evaluating the Decision Tree Model

Let's train a decision tree model and evaluate its performance.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_processed, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test_processed)

# Evaluate the model
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_predictions)

print("Decision Tree Model Evaluation:")
print(f"Root Mean Squared Error: £{dt_rmse:,.2f}")
print(f"R-squared Score: {dt_r2:.4f}")
```

[Include output from the above code here]

---

## Training and Evaluating the Random Forest Model

Now, let's train a random forest model and compare its performance.

```python
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test_processed)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Model Evaluation:")
print(f"Root Mean Squared Error: £{rf_rmse:,.2f}")
print(f"R-squared Score: {rf_r2:.4f}")
```

[Include output from the above code here]

---

## Model Comparison and Interpretation

Let's compare the performance of both models and interpret their results.

```python
print("\nModel Comparison:")
print(f"Decision Tree: RMSE = £{dt_rmse:,.2f}, R2 = {dt_r2:.4f}")
print(f"Random Forest: RMSE = £{rf_rmse:,.2f}, R2 = {rf_r2:.4f}")

# Feature importance
feature_names = numeric_features + pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names(categorical_features).tolist()

# Decision Tree feature importance
dt_importances = dt_model.feature_importances_
dt_indices = np.argsort(dt_importances)[::-1]

# Random Forest feature importance
rf_importances = rf_model.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]

# Plot comparison of top 10 features
plt.figure(figsize=(12,6))
plt.title('Top 10 Feature Importances: Decision Tree vs Random Forest')
x = range(10)
plt.bar(x, dt_importances[dt_indices][:10], width=0.4, align='center', label='Decision Tree')
plt.bar([i+0.4 for i in x], rf_importances[rf_indices][:10], width=0.4, align='center', label='Random Forest')
plt.xticks([i+0.2 for i in x], [feature_names[i] for i in dt_indices][:10], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Visualize a sample decision tree
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=feature_names, filled=True, rounded=True, max_depth=3)
plt.show()

# Residual analysis
plt.figure(figsize=(10,6))
plt.scatter(rf_predictions, y_test - rf_predictions)
plt.title('Residual Plot (Random Forest)')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

[Include the following plots here:
1. Top 10 Feature Importances: Decision Tree vs Random Forest
2. Visualized sample decision tree
3. Residual Plot (Random Forest)]

[Include output from the model comparison here]

---

## Advanced Techniques

Here are some advanced techniques to further improve our models:

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X_train_processed, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print(f"Cross-validation RMSE scores: {cv_rmse}")
print(f"Mean CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")
```

[Include output from the above code here]

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_processed, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))
```

[Include output from the above code here]

### Feature Engineering

Consider creating new features or transforming existing ones:

```python
# Log transform of price (target variable)
y_log = np.log(df['Price'])

# Interaction terms
df['Area_per_Room'] = df['Area in sq ft'] / (df['No. of Bedrooms'] + df['No. of Bathrooms'] + df['No. of Receptions'])

# You would then need to rerun the modeling process with these new features
```

---

## Conclusion

In this lesson, we've explored the use of Decision Trees and Random Forests for predicting house prices in London. We've covered:

- Data loading, cleaning, and exploration
- Feature engineering and preprocessing
- Training and evaluating Decision Tree and Random Forest models
- Model interpretation through feature importance and visualization
- Advanced techniques like cross-validation and hyperparameter tuning

Key takeaways:
1. Random Forests generally outperform single Decision Trees due to their ensemble nature.
2. Feature importance helps us understand which factors most influence house prices.
3. Location and property characteristics play significant roles in determining house prices.
4. Advanced techniques like cross-validation and hyperparameter tuning can further improve model performance.

---

## Further Reading

1. "Introduction to Machine Learning with Python" by Andreas C. Müller and Sarah Guido
2. Scikit-learn Documentation on Decision Trees and Random Forests: 
   - https://scikit-learn.org/stable/modules/tree.html
   - https://scikit-learn.org/stable/modules/ensemble.html#random-forests
3. "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
4. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman

---

**Next Steps:**

- Experiment with other algorithms like Gradient Boosting (e.g., XGBoost, LightGBM)
- Explore more advanced feature engineering techniques
- Investigate the impact of outliers and consider robust regression techniques
- Deploy the model as a simple web application for real-time predictions

Remember, the goal is not just to create accurate predictions, but also to gain insights into the factors influencing house prices in London. Happy modeling!