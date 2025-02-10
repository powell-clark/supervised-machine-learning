## Feature Scaling: A Preview

Feature scaling is a crucial preprocessing step in many machine learning workflows. While we won't need it for our initial decision tree model (as decision trees are invariant to monotonic transformations), we'll revisit this concept when we compare different models later, particularly for:

- Logistic regression
- Support vector machines (SVM)
- Neural networks
- K-nearest neighbours (KNN)

The main scaling methods we'll explore later include:
1. Standardisation (Standard Scaling)
2. Normalisation (Min-Max Scaling)
3. Robust Scaling
4. Log Transformation

### Why Not Scale Now?

Decision trees have several properties that make them robust without scaling:
- They make splits based on relative ordering, not absolute values
- They're invariant to monotonic transformations
- They don't assume any particular distribution of the data

## Feature Scaling

Feature scaling is the process of transforming numerical features to a standard scale. While not strictly necessary for decision trees (as they're invariant to monotonic transformations), understanding scaling methods is crucial for:

1. Data exploration and visualisation
2. Using the same dataset with different algorithms
3. Comparing feature importances across different scales
4. Improving model convergence with certain algorithms

Let's explore the main scaling methods:

#### 1. Standardisation (Standard Scaling)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**What it does:**
- Scales data to have mean = 0 and standard deviation = 1
- Formula: z = (x - μ) / σ

**When to use:**
- When data is approximately normally distributed
- With linear models, logistic regression, neural networks
- When outliers should have less impact
- When comparing features that have different scales

**Pros:**
- Makes features comparable on same scale
- Handles outliers better than min-max scaling
- Required for many machine learning algorithms
- Preserves useful information about outliers

**Cons:**
- Doesn't guarantee bounded range
- May not be suitable for sparse data
- Assumes normal distribution
- Less interpretable than original values

#### 2. Normalisation (Min-Max Scaling)
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**What it does:**
- Scales data to a fixed range, usually [0, 1]
- Formula: z = (x - min(x)) / (max(x) - min(x))

**When to use:**
- When you need values in a bounded range [0,1]
- With neural networks
- When data is not normally distributed
- When outliers are meaningful

**Pros:**
- Preserves zero values
- Handles outliers better than standardisation
- Preserves shape of original distribution
- Good for image processing and neural networks

**Cons:**
- Doesn't handle outliers well if you want to diminish their impact
- Doesn't make data more gaussian-like
- Different features might still have different variances

#### 3. Robust Scaling
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

**What it does:**
- Scales using statistics that are robust to outliers
- Uses median and interquartile range instead of mean and variance

**When to use:**
- When data has significant outliers
- When you want to preserve shape for non-normal distributions
- With small datasets where outlier impact is significant

**Pros:**
- Robust to outliers
- Preserves shape for non-normal distributions
- Good for small datasets
- Doesn't assume normal distribution

**Cons:**
- Might not scale data as precisely as other methods
- Can be computationally more expensive
- May not be necessary for large datasets
- Less common, so less supported in some libraries

 #### 4. Log Transform
 ```python
 import numpy as np
 
 X_logged = np.log1p(X)  # log1p is log(1+x), handles zeros better
 ```
 
 **What it does:**
 - Transforms data using natural logarithm
 - Reduces skewness and makes data more normal-like
 - Compresses high values while spreading out low values
 
 **When to use:**
 - With highly skewed data
 - When data has multiplicative relationships
 - With power-law distributions
 - When dealing with prices or populations
 
 **Pros:**
 - Makes highly skewed data more normal-like
 - Reduces impact of outliers
 - Preserves relative differences
 - Useful for financial data
 
 **Cons:**
 - Can only be used with positive values
 - Makes interpretation less intuitive
 - May over-compress large values
 - Not reversible without loss of precision
 
 #### 5. Box-Cox Transform
 ```python
 from scipy import stats
 
 X_boxcox = stats.boxcox(X)  # Automatically finds optimal lambda
 ```
 
 **What it does:**
 - Family of power transformations
 - Finds optimal transformation parameter (lambda)
 - Special case: lambda=0 is log transform
 
 **When to use:**
 - When data needs to be more normal-like
 - With positive, continuous data
 - When relationship between variables is non-linear
 
 #### 6. Yeo-Johnson Transform
 ```python
 from sklearn.preprocessing import PowerTransformer
 
 pt = PowerTransformer(method='yeo-johnson')
 X_yeojohnson = pt.fit_transform(X)
 ```
 
 **What it does:**
 - Similar to Box-Cox but works with negative values
 - Automatically finds optimal transformation
 - Aims to make data more normal-like
 
 **When to use:**
 - When data includes negative values
 - When normality is important
 - With continuous variables

The choice of scaling method should be based on:
- The distribution of your data
- The requirements of your algorithms
- The importance of interpretability
- The presence and significance of outliers

### Decision Trees and Scaling

For our house price prediction task using decision trees, scaling wasn't strictly necessary because:

1. Decision trees make splits based on relative ordering, not absolute values
2. They're invariant to monotonic transformations of individual features
3. They don't assume any particular distribution of the data

However, scaling might still be useful for:

1. Visualising the data
2. Comparing feature importances
3. Using the same data with other algorithms
4. Improving numerical stability in some cases

For our dataset, which is not normally distributed (particularly house prices), we might want to consider:
- Log transformation for price values (common in real estate)
- Robust scaling if we want to reduce the impact of outliers
- Keeping original values for interpretability


### Comparing Linear Regression with Decision Trees and Random Forests

Let's compare our decision tree and random forest models to a linear regression model (not logistic regression, as we're dealing with a continuous price prediction problem, not classification).

To do this we will evaluate the distribution of our data and then apply appropriate scaling methods to able to compare the results between the models.

We'll explore df_filled_unknown in this example as it has no missing values and doesn't introduce dimensionality problems (as we're not one-hot encoding the location) which would greatly effect the performance of the linear regression model.

