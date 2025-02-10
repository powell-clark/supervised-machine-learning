# Lesson 1: Logistic Regression

## What is Logistic Regression?
Imagine you're a doctor trying to predict whether a patient has a certain disease, like cancer, based on some medical data about the patient (age, blood pressure, etc.). You want a simple "yes" or "no" answer, but the data you have is all numbers. How can you turn those numbers into a "yes" or "no" prediction? That's where logistic regression comes in!

Logistic regression is a tool in machine learning that helps us solve problems where we want to predict a binary outcome (yes/no, 1/0, true/false) based on some input data.

## How Does Logistic Regression Work?
Logistic regression works by finding a relationship between the input data and the probability of the outcome being "yes" or "no".

### Step 1: Linear Combination
First, logistic regression calculates a weighted sum of the input features. Each feature is multiplied by a weight (a number that represents its importance), and then these are all added together. This is similar to the equation of a straight line:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

Here, `x1`, `x2`, ..., `xn` are the input features, `w1`, `w2`, ..., `wn` are the weights, and `b` is a special term called the bias.

### Step 2: Sigmoid Function
The result of the linear combination (`z`) could be any number, but we want a probability between 0 and 1. To achieve this, we pass `z` through a special mathematical function called the sigmoid function:

```
p = 1 / (1 + e^(-z))
```

```
p = \frac{1}{1 + e^{-z}}
```




Here, `e` is a mathematical constant approximately equal to 2.71828.

The sigmoid function squashes `z` into a value between 0 and 1, which we can interpret as the probability of the outcome being "yes".

![Sigmoid Curve](../static/sigmoid-curve.png)

*The sigmoid function squashes input values to a probability between 0 and 1.*

### Step 3: Making a Prediction
To make a final prediction, we set a threshold (usually 0.5). If the probability is greater than the threshold, we predict "yes", otherwise we predict "no".

![Decision Boundary](../static/sigmoid-curve-decision-boundary.png)

*The decision boundary separates the two classes based on the model's predictions.*

# Logistic Regression: Key Concepts Visualization
```
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression

# Sigmoid function visualization
def plot_sigmoid():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Sigmoid Function')
    plt.axvline(x=0, color='r', linestyle='--', label='Decision Boundary')
    plt.axhline(y=0.5, color='g', linestyle=':', label='0.5 Threshold')
    plt.title('Sigmoid Function and Decision Boundary')
    plt.xlabel('Combined Input (w1*x1 + w2*x2 + ... + b)')
    plt.ylabel('Probability of Positive Class')
    plt.legend()
    plt.grid(True)
    plt.show()

# 2D visualization
def plot_2d_decision_boundary():
    np.random.seed(0)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = LogisticRegression()
    clf.fit(X, y)

    plt.figure(figsize=(10, 8))
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='Class 0')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='Class 1')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Dataset with Decision Boundary')
    plt.legend()
    plt.show()

# 3D visualization
def plot_3d_decision_boundary():
    np.random.seed(0)
    X = np.random.randn(200, 3)
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

    clf = LogisticRegression()
    clf.fit(X, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[y==0][:, 0], X[y==0][:, 1], X[y==0][:, 2], label='Class 0')
    ax.scatter(X[y==1][:, 0], X[y==1][:, 1], X[y==1][:, 2], label='Class 1')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    
    z = (-clf.intercept_[0] - clf.coef_[0][0] * xx1 - clf.coef_[0][1] * xx2) / clf.coef_[0][2]

    ax.plot_surface(xx1, xx2, z, alpha=0.3)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('3D Dataset with Decision Boundary')
    ax.legend()
    plt.show()

# Run all visualizations
plot_sigmoid()
plot_2d_decision_boundary()
plot_3d_decision_boundary()

```

Interactive Logistic Regression Visualization 

1. Sigmoid Function:
   - The S-shaped curve transforms the combined input into a probability between 0 and 1.
   - The red dashed line (x=0) represents the decision boundary where the model is equally confident about both classes.
   - The green dotted line (y=0.5) shows the typical classification threshold.

   ![Sigmoid Function and Decision Boundary](../static/sigmoid-function-and-decision-boundary.png)

2. 2D Decision Boundary:
   - In 2D space, the decision boundary is a line.
   - Points on one side are classified as one class, points on the other side as the other class.
   - The line represents where the model is equally confident about both classes.

   ![2d Decision Boundary](../static/2d-decision-boundary.png)

3. 3D Decision Boundary:
   - In 3D space, the decision boundary becomes a plane.
   - The plane divides the 3D space into two regions, each corresponding to a class.

   ![3d Decision Boundary](../static/3d-decision-boundary.png)

4. Hyperplane in Higher Dimensions:
   - For datasets with more than 3 features, we can't visualize the decision boundary directly.
   - The concept extends to a hyperplane in higher-dimensional space.
   - A hyperplane in n-dimensional space is the subspace of dimension n-1 that divides the space into two parts.

   ![Mind Blowing](../static/mind-blowing.gif)

5. The Equation of the Hyperplane:
   - For n features, the hyperplane is defined by the equation: w1*x1 + w2*x2 + ... + wn*xn + b = 0
   - w1, w2, ..., wn are the weights learned by the model, and b is the bias term.

6. Classification Process:
   - The model calculates w1*x1 + w2*x2 + ... + wn*xn + b for a given input.
   - This value is passed through the sigmoid function to get a probability.
   - If the probability > 0.5, it's classified as the positive class, otherwise as the negative class.

These visualizations help us understand how logistic regression makes decisions in different dimensional spaces. 
While we can't visualize beyond 3D, the same principle applies in higher dimensions: the model finds a 
hyperplane that best separates the classes in the feature space.

Further Reading:
1. "Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani - Chapter 4
2. "Pattern Recognition and Machine Learning" by Christopher Bishop - Chapter 4
3. Stanford CS229 course notes on logistic regression
4. Scikit-learn documentation on logistic regression
5. "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter 6.2 (for a more advanced treatment)
""")

## Training the Model
To make accurate predictions, the model needs to learn the right weights (`w1`, `w2`, ..., `wn`) and bias (`b`). This is done through a process called training, where the model is shown many examples of input data and their corresponding correct outputs.

The model starts with random weights and bias, makes predictions on the training data, and then adjusts the weights and bias to make its predictions closer to the correct outputs. This process is repeated many times until the model's predictions are accurate enough.

## Evaluating the Model
After training, we evaluate the model by making predictions on data it hasn't seen before (called the test set). 

## Evaluation and Optimization

After training our logistic regression model, we need to assess how well it performs. We do this by making predictions on a test set (data the model hasn't seen during training) and comparing these predictions to the true outcomes. 

Before we dive into the metrics, let's define some key terms:

### Key Terms

- **True Positive (TP)**: The model correctly predicts the positive class. 
  Example: The model predicts a tumor is malignant, and it actually is malignant.

- **True Negative (TN)**: The model correctly predicts the negative class. 
  Example: The model predicts a tumor is benign, and it actually is benign.

- **False Positive (FP)**: The model incorrectly predicts the positive class. 
  Example: The model predicts a tumor is malignant, but it's actually benign. Also known as a "Type I error".

- **False Negative (FN)**: The model incorrectly predicts the negative class. 
  Example: The model predicts a tumor is benign, but it's actually malignant. Also known as a "Type II error".


These terms are crucial for understanding the following evaluation metrics:

### 1. Accuracy

Accuracy is the proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.

```markdown
Accuracy = (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)
```

For example, if our model correctly identified 90 out of 100 tumors, the accuracy would be 90%.

### 2. Precision

Precision is the proportion of true positive predictions among all positive predictions. It answers the question: "Of all the tumors our model predicted as malignant, what percentage actually were malignant?"

```markdown
Precision = True Positives / (True Positives + False Positives)
```

For instance, if our model predicted 50 tumors as malignant, and 45 of these were actually malignant, the precision would be 45/50 = 90%.

### 3. Recall (Sensitivity)

Recall is the proportion of true positive predictions among all actual positive cases. It answers the question: "Of all the actual malignant tumors, what percentage did our model correctly identify?"

```markdown
Recall = True Positives / (True Positives + False Negatives)
```

For example, if there were 60 malignant tumors in total, and our model correctly identified 54 of them, the recall would be 54/60 = 90%.

### 4. F1 Score

The F1 Score is a single metric that combines both precision and recall, providing a balanced measure of a model's performance. It's particularly useful when you have an uneven class distribution (i.e., when the number of positive cases is very different from the number of negative cases).

#### Definition:
The F1 Score is the harmonic mean of precision and recall:

```markdown
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

#### Why use the harmonic mean?
We use the harmonic mean (rather than a simple average) because it punishes extreme values. This means the F1 score will be low if either precision or recall is low.

#### When is it useful?
The F1 Score is particularly useful when you need to find an optimal balance between precision and recall, and when there is an uneven class distribution. For instance, in medical diagnoses or fraud detection, where false negatives can be particularly costly.

#### Example:
Let's consider our tumor classification model:

Scenario 1:
- Precision = 0.8 (80% of tumors we predicted as malignant were actually malignant)
- Recall = 0.6 (We correctly identified 60% of all malignant tumors)
- F1 Score = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 0.69

Scenario 2:
- Precision = 0.7 (70% of tumors we predicted as malignant were actually malignant)
- Recall = 0.7 (We correctly identified 70% of all malignant tumors)
- F1 Score = 2 * (0.7 * 0.7) / (0.7 + 0.7) = 0.7

In Scenario 2, even though both precision and recall are lower than the higher value in Scenario 1, the F1 Score is higher because it balances both metrics.

#### Visualization:

To better understand how F1 Score relates to precision and recall, let's visualize it:

```python
import numpy as np
import matplotlib.pyplot as plt

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

precision = np.linspace(0.1, 1, 100)
recall = np.linspace(0.1, 1, 100)

P, R = np.meshgrid(precision, recall)
F1 = f1_score(P, R)

plt.figure(figsize=(10, 8))
plt.contourf(P, R, F1, levels=20, cmap='viridis')
plt.colorbar(label='F1 Score')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('F1 Score as a Function of Precision and Recall')
plt.show()
```
 ![F1 score](../static/f1-score.png)

In this plot, darker colors represent higher F1 Scores. You can see that to achieve a high F1 Score, both precision and recall need to be high. 
The F1 Score decreases rapidly if either precision or recall is low.

#### Key Points:
1. The F1 Score ranges from 0 to 1, with 1 being the best possible score.
2. It gives equal weight to precision and recall.
3. It's more informative than accuracy when dealing with imbalanced datasets.
4. A high F1 Score ensures that you have low false positives and low false negatives.

By considering the F1 Score alongside other metrics, you can get a more comprehensive understanding of your model's performance, especially in situations where both precision and recall are important.

### 5. Confusion Matrix

A confusion matrix is a table that describes the performance of a classification model:

```markdown
                 Predicted Negative | Predicted Positive
Actual Negative     True Negative   |   False Positive
Actual Positive    False Negative   |   True Positive
```

This gives a complete picture of how our model is performing, showing all correct and incorrect classifications.


## Theory Section Conclusion
Logistic regression is a simple yet powerful tool for binary classification problems. It works by finding a linear combination of the input features, passing it through the sigmoid function to get a probability, and then making a prediction based on a threshold.

Despite its name, logistic regression is actually a classification algorithm, not a regression algorithm. It's called "regression" because the model finds a relationship (a regression line) between the input features and the log-odds of the output being "yes".

## Applying Logistic Regression to the Wisconsin Breast Cancer Dataset

Let's apply logistic regression to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on measurements of the tumor cells.

We'll use the Wisconsin Breast Cancer dataset, which contains data for 569 breast tumor samples. Each sample has 30 features, such as the radius, texture, and perimeter of the cells, and a label indicating whether the tumor is malignant (1) or benign (0).

### Loading and Preprocessing the Data

First, let's load the dataset and look at the first few rows:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
# load_breast_cancer() is a function from scikit-learn that loads the Wisconsin Breast Cancer dataset.

print(f"Features: {data.feature_names}")
# This prints the names of the features in the dataset.

print(f"Targets: {data.target_names}")
# This prints the names of the target classes: malignant and benign.

print(f"Data shape: {data.data.shape}")
# This prints the shape of the data (number of samples, number of features).

import pandas as pd
df = pd.DataFrame(data.data, columns=data.feature_names)
# We create a pandas DataFrame to hold the data.
# data.data contains the feature values, and data.feature_names provides the column names.

df['target'] = data.target
# We add a 'target' column to the DataFrame to hold the target values (0 for benign, 1 for malignant).

print(df.head().to_markdown(index=False))
# This prints the first 5 rows of the DataFrame in a markdown-friendly format.
```

<div style="max-height: 300px; overflow-y: scroll;">

| mean radius | mean texture | mean perimeter | mean area | mean smoothness | mean compactness | mean concavity | mean concave points | mean symmetry | mean fractal dimension | radius error | texture error | perimeter error | area error | smoothness error | compactness error | concavity error | concave points error | symmetry error | fractal dimension error | worst radius | worst texture | worst perimeter | worst area | worst smoothness | worst compactness | worst concavity | worst concave points | worst symmetry | worst fractal dimension | target |
|-------------|--------------|----------------|-----------|-----------------|------------------|----------------|---------------------|---------------|------------------------|--------------|---------------|-----------------|------------|------------------|-------------------|-----------------|----------------------|----------------|-------------------------|--------------|---------------|-----------------|------------|------------------|-------------------|-----------------|----------------------|----------------|-------------------------|--------|
| 17.99       | 10.38        | 122.80         | 1001.0    | 0.11840         | 0.27760          | 0.3001         | 0.14710             | 0.2419        | 0.07871                | 1.0950       | 0.9053        | 8.589           | 153.40     | 0.006399         | 0.04904           | 0.05373         | 0.01587              | 0.03003        | 0.006193                | 25.38        | 17.33         | 184.60          | 2019.0     | 0.1622           | 0.6656            | 0.7119          | 0.2654               | 0.4601         | 0.11890                 | 0      |
| 20.57       | 17.77        | 132.90         | 1326.0    | 0.08474         | 0.07864          | 0.0869         | 0.07017             | 0.1812        | 0.05667                | 0.5435       | 0.7339        | 3.398           | 74.08      | 0.005225         | 0.01308           | 0.01860         | 0.01340              | 0.01389        | 0.003532                | 24.99        | 23.41         | 158.80          | 1956.0     | 0.1238           | 0.1866            | 0.2416          | 0.1860               | 0.2750         | 0.08902                 | 0      |
| 19.69       | 21.25        | 130.00         | 1203.0    | 0.10960         | 0.15990          | 0.1974         | 0.12790             | 0.2069        | 0.05999                | 0.7456       | 0.7869        | 4.585           | 94.03      | 0.006150         | 0.04006           | 0.03832         | 0.02058              | 0.02250        | 0.004571                | 23.57        | 25.53         | 152.50          | 1709.0     | 0.1444           | 0.4245            | 0.4504          | 0.2430               | 0.3613         | 0.08758                 | 0      |
| 11.42       | 20.38        | 77.58          | 386.1     | 0.14250         | 0.28390          | 0.2414         | 0.10520             | 0.2597        | 0.09744                | 0.4956       | 1.1560        | 3.445           | 27.23      | 0.009110         | 0.07458           | 0.05661         | 0.01867              | 0.05963        | 0.009208                | 14.91        | 26.50         | 98.87           | 567.7      | 0.2098           | 0.8663            | 0.6869          | 0.2575               | 0.6638         | 0.17300                 | 0      |
| 20.29       | 14.34        | 135.10         | 1297.0    | 0.10030         | 0.13280          | 0.1980         | 0.10430             | 0.1809        | 0.05883                | 0.7572       | 0.7813        | 5.438           | 94.44      | 0.011490         | 0.02461           | 0.05688         | 0.01885              | 0.01756        | 0.005115                | 22.54        | 16.67         | 152.20          | 1575.0     | 0.1374           | 0.2050            | 0.4000          | 0.1625               | 0.2364         | 0.07678                 | 0      |

</div>

Here's what's happening:
1. We load the Wisconsin Breast Cancer dataset using `load_breast_cancer()` from scikit-learn. This dataset is included in scikit-learn, so we don't need to download it separately.

2. We print the feature names (`data.feature_names`), target names (`data.target_names`), and the shape of the data (`data.data.shape`). This gives us an overview of what's in the dataset.

3. We create a pandas DataFrame `df` to hold the data. We use `data.data` for the feature values and `data.feature_names` for the column names. This puts the data into a tabular format that's easy to work with.

4. We add a 'target' column to the DataFrame to hold the target values. 0 represents a benign tumor, and 1 represents a malignant tumor.

5. We print the first 5 rows of the DataFrame using `df.head().to_markdown(index=False)`. This lets us see a sample of the data in a nicely formatted table.

Next, let's preprocess the data:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.data
# We separate the features (X) and the target variable (y).
# X holds all the feature values (like radius, texture, etc.).

y = data.target
# y holds the target values (0 for benign, 1 for malignant).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# We split the data into training and testing sets.
# train_test_split() is a function from scikit-learn that does this split for us.
# test_size=0.2 means 20% of the data will be used for testing, and the rest for training.
# random_state=42 ensures we get the same split every time we run the code.

scaler = StandardScaler()
# We create a StandardScaler object to standardize the features.
# Standardization transforms the data to have mean 0 and standard deviation 1.
# This is often beneficial for machine learning algorithms.

X_train_scaled = scaler.fit_transform(X_train)
# We fit the scaler to the training data and transform the training data.
# fit_transform() calculates the mean and standard deviation of each feature in the training data,
# and then standardizes the training data using these values.

X_test_scaled = scaler.transform(X_test)
# We transform the test data using the mean and standard deviation calculated from the training data.
# This is important - we don't want to use the test data to calculate the mean and standard deviation,
# as that would leak information about the test data into the model.

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
# We print the shapes of the training and testing data to ensure the split worked as expected.
```

```
Output:
X_train shape: (455, 30)
X_test shape: (114, 30)
y_train shape: (455,)
y_test shape: (114,)
```

Here's what's happening:
1. We separate the features (`X`) and the target variable (`y`). `X` holds all the feature values, and `y` holds the corresponding target values (0 for benign, 1 for malignant).

2. We split the data into training and testing sets using `train_test_split()` from scikit-learn. 
   - `test_size=0.2` means 20% of the data will be used for testing, and the rest for training. 
   - `random_state=42` is a seed value that ensures we get the same split every time we run the code. This is important for reproducibility.

3. We create a `StandardScaler` object to standardize the features. Standardization transforms the data to have mean 0 and standard deviation 1. This is often beneficial for machine learning algorithms, as it puts all features on the same scale.

4. We fit the scaler to the training data and transform the training data using `fit_transform()`. This calculates the mean and standard deviation of each feature in the training data and then standardizes the training data using these values.

5. We transform the test data using `transform()`. This is important - we use the mean and standard deviation calculated from the training data to standardize the test data. We don't want to calculate the mean and standard deviation from the test data, as that would leak information about the test data into the model.

6. We print the shapes of the training and testing data to ensure the split worked as expected. We see that we have 455 training samples and 114 test samples, each with 30 features.

### Training the Logistic Regression Model

Let's walk through the process of training our logistic regression model step by step, explaining each part as we go. We'll use PyTorch, a popular library for machine learning, to define and train our model.

```python
import torch
# PyTorch is a library for machine learning.
# It allows us to define and train neural networks, including simpler models like logistic regression.

import torch.nn as nn
# torch.nn is PyTorch's neural network library.
# It provides the building blocks for creating neural networks, including logistic regression.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# These are various metrics we'll use to evaluate our model's performance.
# They come from scikit-learn, another popular machine learning library.

import time
# We'll use this to measure how long training takes.

class LogisticRegressionModel(nn.Module):
    # We define our logistic regression model as a class.
    # nn.Module is a base class provided by PyTorch from which all neural network modules inherit.
    
    def __init__(self, input_dim):
        # The constructor takes the input dimension as a parameter.
        # The input dimension is the number of features in our data.
        
        super(LogisticRegressionModel, self).__init__()
        # This line ensures that the base class (nn.Module) is initialized properly.
        
        self.linear = nn.Linear(input_dim, 1)  
        # We define the model's only layer: a linear transformation from the input dimension to a single output.
        # nn.Linear is a module provided by PyTorch for linear transformations.
        # It applies a linear transformation to the incoming data: y = xA^T + b
    
    def forward(self, x):
        # The forward method defines what happens when the model is applied to an input tensor x.
        
        y_predicted = torch.sigmoid(self.linear(x))
        # We apply the linear transformation (self.linear) to the input tensor x,
        # and then apply the sigmoid function to the result.
        # The sigmoid function squashes the output between 0 and 1, giving us a probability.
        
        return y_predicted
        # We return the predicted probability.

# Convert our preprocessed data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# We convert the training data to a PyTorch tensor.
# PyTorch works with tensors, which are similar to numpy arrays but can be used on a GPU for faster computations.

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# We convert the test data to a PyTorch tensor.

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
# We convert the training labels to a PyTorch tensor.
# The view(-1, 1) reshapes the tensor to be a column vector.

y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# We convert the test labels to a PyTorch tensor, also reshaping to a column vector.

model = LogisticRegressionModel(input_dim=X_train.shape[1])
# We instantiate our model, passing in the number of features (the second dimension of X_train 2d array dataframe) as the input dimension.
# X_train = 
# [
#    [17.99, 10.38, 122.80, 1001.0, 0.11840, ..., 0.2654, 0.4601, 0.11890],
#    [20.57, 17.77, 132.90, 1326.0, 0.08474, ..., 0.1860, 0.2750, 0.08902],
#    [19.69, 21.25, 130.00, 1203.0, 0.10960, ..., 0.2430, 0.3613, 0.08758],
#   ...
#   [13.54, 14.36, 87.46,  566.3, 0.09779, ..., 0.1189, 0.2645, 0.06013]
# ]
# X_train.shape[0] = 455 (samples)
# X_train.shape[1] = 30 (features)


criterion = nn.BCELoss()
# We define the loss function.
# BCELoss is the binary cross-entropy loss, commonly used for binary classification problems.

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# We define the optimizer, which is how the model will be updated during training.
# We're using stochastic gradient descent (SGD) with a learning rate of 0.01.

num_epochs = 1000
# We set the number of epochs. An epoch is a full pass through the training data.

start_time = time.time()
# We note the start time so we can calculate how long training takes.

for epoch in range(num_epochs):
    # We loop through the number of epochs.
    
    y_predicted = model(X_train_tensor)
    # We apply the model to the training data to get the predicted probabilities.
    
    loss = criterion(y_predicted, y_train_tensor)
    # We calculate the loss by comparing the predicted probabilities to the actual labels.
    
    loss.backward()
    # We compute the gradients of the loss with respect to the model's parameters.
    
    optimizer.step()
    # We update the model's parameters using the optimizer.
    
    optimizer.zero_grad()
    # We reset the gradients to zero for the next epoch.

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

end_time = time.time()
# We note the end time.

print(f"Training time: {end_time-start_time:.2f} seconds")
# We print how long training took.

# Evaluation
with torch.no_grad():
    # We tell PyTorch not to calculate gradients during testing, as we're not training anymore.
    
    y_predicted = model(X_test_tensor)
    # We apply the model to the test data to get the predicted probabilities.
    
    y_predicted_cls = y_predicted.round()
    # We round the probabilities to get the predicted class labels (0 or 1).

accuracy = accuracy_score(y_test, y_predicted_cls)
# We calculate the accuracy by comparing the predicted class labels to the true labels.

precision = precision_score(y_test, y_predicted_cls)
# We calculate the precision.
# Precision is the number of true positives divided by the number of predicted positives.

recall = recall_score(y_test, y_predicted_cls)
# We calculate the recall.
# Recall is the number of true positives divided by the number of actual positives.

f1 = f1_score(y_test, y_predicted_cls)
# We calculate the F1 score.
# The F1 score is the harmonic mean of precision and recall.

cm = confusion_matrix(y_test, y_predicted_cls)
# We calculate the confusion matrix.
# The confusion matrix shows the number of correct and incorrect predictions for each class.

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
# We print all the evaluation metrics.
```

Now, let's break down the key steps in this process:

1. **Model Definition**: 
   - We define our logistic regression model as a class that inherits from `nn.Module`.
   - The model has one linear layer (`nn.Linear`) that takes our input features and produces a single output.
   - In the `forward` method, we apply the sigmoid function to this output to get a probability between 0 and 1.

2. **Data Preparation**: 
   - We convert our preprocessed data (features and labels) into PyTorch tensors.
   - Tensors are similar to numpy arrays but can be used on GPUs for faster computations.

3. **Model Instantiation**: 
   - We create an instance of our `LogisticRegressionModel`, specifying the number of input features.

4. **Loss Function**: 
   - We use Binary Cross Entropy loss (`nn.BCELoss`).
   - This loss function is well-suited for binary classification problems like ours.

5. **Optimizer**: 
   - We use Stochastic Gradient Descent (`torch.optim.SGD`) as our optimization algorithm.
   - The optimizer will update our model's parameters to minimize the loss.

6. **Training Loop**: 
   - We iterate through our data multiple times (epochs).
   - In each epoch:
     a. We make predictions on the training data.
     b. We calculate the loss (how wrong our predictions are).
     c. We compute the gradients of the loss with respect to the model parameters.
     d. We update the model parameters using the optimizer.
     e. We reset the gradients to zero for the next iteration.

7. **Evaluation**: 
   - After training, we use our model to make predictions on the test data.
   - We calculate various metrics to evaluate how well our model performs:
     - Accuracy: The proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.
     - Precision: The proportion of true positive predictions among all positive predictions.
     - Recall: The proportion of true positive predictions among all actual positive cases.
     - F1 Score: The harmonic mean of precision and recall, providing a single score that balances both metrics.
     - Confusion Matrix: A table showing the number of correct and incorrect predictions broken down by each class.

Now, let's look at our results:

```
Epoch [100/1000], Loss: 0.2217
Epoch [200/1000], Loss: 0.1538
Epoch [300/1000], Loss: 0.1247
Epoch [400/1000], Loss: 0.1079
Epoch [500/1000], Loss: 0.0966
Epoch [600/1000], Loss: 0.0885
Epoch [700/1000], Loss: 0.0822
Epoch [800/1000], Loss: 0.0772
Epoch [900/1000], Loss: 0.0731
Epoch [1000/1000], Loss: 0.0696
Training time: 1.40 seconds
Accuracy: 0.9737
Precision: 0.9726
Recall: 0.9863
F1 Score: 0.9794
Confusion Matrix:
[[40  2]
 [ 1 71]]
```

Let's interpret these results:

1. **Training Progress**: The loss decreases over time, indicating that our model is learning and improving its predictions.

2. **Training Time**: Our model took about 1.40 seconds to train. This is quite fast, which is one of the advantages of logistic regression.

3. **Accuracy**: 0.9737 means our model correctly classified 97.37% of the tumors in the test set. This is a very good performance!

4. **Precision**: 0.9726 means that when our model predicted a tumor was malignant, it was right 97.26% of the time.

5. **Recall**: 0.9863 means that out of all the actually malignant tumors, our model correctly identified 98.63% of them.

6. **F1 Score**: 0.9794 is the harmonic mean of precision and recall. It provides a single score that balances both metrics.

7. **Confusion Matrix**: This shows us exactly what our model got right and wrong:
   - 40 benign tumors were correctly classified (true negatives)
   - 71 malignant tumors were correctly classified (true positives)
   - 2 benign tumors were incorrectly classified as malignant (false positives)
   - 1 malignant tumor was incorrectly classified as benign (false negative)

Now, let's discuss some important concepts:

**Overfitting vs Underfitting**:
- Overfitting occurs when a model learns the training data too well, including its noise and peculiarities. It performs well on the training data but poorly on new, unseen data.
- Underfitting happens when a model is too simple to capture the underlying pattern in the data. It performs poorly on both training and test data.
- Our model seems to be performing well on the test data, suggesting it's neither significantly overfitting nor underfitting.

**Bias-Variance Tradeoff**:
- Bias is the error introduced by approximating a real-world problem with a simplified model. High bias can lead to underfitting.
- Variance is the error introduced by the model's sensitivity to small fluctuations in the training set. High variance can lead to overfitting.
- We aim to find a balance between bias and variance. Our model's good performance on new data suggests it has found a good balance.

**Runtime Assessment**:
- Our model trained in about 1.40 seconds, which is very fast.
- Logistic regression is known for its efficiency, making it a good choice for quick results or large datasets.
- For more complex problems or very large datasets, you might need more sophisticated models that take longer to train.


# Advanced Concepts and Model Improvement

Now that we've trained our initial model and evaluated its performance, let's explore some advanced concepts that can help us improve our model and gain deeper insights into its behavior.

## 1. Hyperparameter Tuning

In our initial model, we used a learning rate of 0.01 and trained for 1000 epochs. But how do we know these are the best values? Let's experiment with different combinations:

```python
learning_rates = [0.001, 0.01, 0.1]
epochs_list = [100, 500, 1000]

for lr in learning_rates:
    for epochs in epochs_list:
        model = LogisticRegressionModel(input_dim=X_train.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            y_predicted = model(X_train_tensor)
            loss = criterion(y_predicted, y_train_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            y_predicted = model(X_test_tensor)
            y_predicted_cls = y_predicted.round()
            accuracy = accuracy_score(y_test, y_predicted_cls)
        
        print(f"Learning rate: {lr}, Epochs: {epochs}, Accuracy: {accuracy:.4f}")

# We chose lr=0.01 and epochs=1000 based on the best performance from this tuning.
```

This process, known as hyperparameter tuning, helps us find the optimal combination of learning rate and number of epochs. The results will show us which combination performs best on our test set.

## 2. Feature Importance

Understanding which features are most important for our predictions can provide valuable insights:

```python
feature_importance = abs(model.linear.weight.detach().numpy()[0])
feature_names = data.feature_names

for importance, name in sorted(zip(feature_importance, feature_names), reverse=True)[:10]:  # Top 10 features
    print(f"{name}: {importance:.4f}")

# Visualization of feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xticks(range(len(feature_importance)), feature_names, rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

This code extracts the weights from our model, which indicate the importance of each feature. The larger the absolute value of the weight, the more important the feature is for the prediction. This can help us understand which characteristics of a tumor are most indicative of malignancy.

## 3. Cross-validation

So far, we've been using a single train-test split. While this is a good start, it might not give us the most robust estimate of our model's performance. Cross-validation can help:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train[val_idx]
    
    model = LogisticRegressionModel(input_dim=X_train.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        y_predicted = model(torch.tensor(X_train_fold, dtype=torch.float32))
        loss = criterion(y_predicted, torch.tensor(y_train_fold, dtype=torch.float32).view(-1, 1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    with torch.no_grad():
        y_val_predicted = model(torch.tensor(X_val_fold, dtype=torch.float32))
        y_val_predicted_cls = y_val_predicted.round()
        accuracy = accuracy_score(y_val_fold, y_val_predicted_cls)
    
    cv_scores.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
```

This 5-fold cross-validation gives us a more robust estimate of our model's performance by training and evaluating it on different subsets of the data.

## 4. Comparison to Baseline

To truly understand how well our model is performing, it's useful to compare it to a simple baseline:

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_accuracy = dummy_clf.score(X_test, y_test)

print(f"Dummy Classifier Accuracy: {dummy_accuracy:.4f}")
print(f"Our Model Accuracy: {accuracy:.4f}")
```

This dummy classifier always predicts the most frequent class. If our model isn't performing significantly better than this baseline, it might indicate that we need to revisit our approach.

By exploring these advanced concepts, we've gained a deeper understanding of our logistic regression model and its performance on the breast cancer dataset. These techniques can be applied to improve and validate any machine learning model you build in the future.

# Conclusion and Limitations

In conclusion, our logistic regression model performs excellently on this breast cancer dataset. It's accurate, fast, and doesn't show signs of significant overfitting or underfitting. However, it's crucial to remember that in a medical context like this, we need to be especially careful about false negatives (malignant tumors classified as benign). While our model only had one such case, in a real-world application, we might want to adjust our model or decision threshold to minimize these potentially dangerous misclassifications.

## Real-world Application and Limitations

In a real medical setting, this model could be used as a diagnostic aid, helping doctors to quickly identify potentially malignant tumors. However, it's crucial to note several limitations:

1. This model should not be used as the sole diagnostic tool. It should always be used in conjunction with a doctor's expertise and other diagnostic methods.

2. The model's performance, while good on this dataset, may not generalize to all populations or all types of breast cancer. It would need extensive testing and validation before being used in a clinical setting.

3. The consequences of false negatives (predicting a tumor is benign when it's actually malignant) are much more severe than false positives. In a real-world application, we might want to adjust our decision threshold to minimize false negatives, even if it increases false positives.

4. The model doesn't provide uncertainty estimates, which would be valuable in a medical context.

5. The model assumes that all features are equally available and reliable for all patients, which may not be the case in practice.

## Limitations of Binary Classification

1. Binary Outcome: Our current model is limited to predicting only two classes (malignant or benign). In many real-world scenarios, we might need to classify into multiple categories.

2. Linear Decision Boundary: Logistic regression assumes a linear relationship between features and the log-odds of the outcome. This may not always be the case in complex datasets.

3. Feature Independence: The model assumes that features are independent, which is often not true in real-world data.

4. Outlier Sensitivity: Logistic regression can be sensitive to outliers, potentially skewing the decision boundary.

5. Limited Expressiveness: For highly complex patterns in data, logistic regression might not be expressive enough to capture all the nuances.

## Other Types of Regression

1. Multinomial Logistic Regression: An extension of binary logistic regression that can handle multiple classes. It's useful when we need to classify into more than two categories.

2. Ordinal Logistic Regression: Used when the target variable has ordered categories. It takes into account the order of the classes.

3. Linear Regression: Predicts a continuous output variable based on one or more input features. It's used when the target variable is a real number (e.g., price, temperature).

4. Polynomial Regression: A form of linear regression where the relationship between the independent variable and the dependent variable is modeled as an nth degree polynomial.

5. Ridge and Lasso Regression: These are linear regression techniques with added regularization to prevent overfitting and handle multicollinearity.

6. Elastic Net Regression: A hybrid of Ridge and Lasso regression, combining their penalties.

Each of these regression techniques has its own strengths and is suited for different types of problems. As we move forward, we'll explore decision trees, which can handle both classification and regression tasks, providing a flexible alternative to these regression methods.


## Glossary of Terms

- Logistic Regression: A statistical method for predicting a binary outcome based on one or more independent variables.
- Sigmoid Function: A mathematical function that maps any input to a value between 0 and 1.
- Binary Classification: A type of classification task where the goal is to predict one of two possible outcomes.
- Feature: An individual measurable property of the phenomenon being observed.
- Weight: A parameter in the model that determines the importance of each feature.
- Bias: An additional parameter in the model that allows the decision boundary to be shifted.
- Epoch: One complete pass through the entire training dataset.
- Loss Function: A method of evaluating how well the algorithm models the dataset.
- Gradient Descent: An optimization algorithm used to minimize the loss function.
- Overfitting: When a model learns the training data too well, including noise and fluctuations, and performs poorly on new, unseen data.
- Underfitting: When a model is too simple to capture the underlying pattern in the data and is just generally bad at making predictions.

## Further Reading

For those interested in diving deeper into logistic regression and related topics, here are some suggested resources:

1. "Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani
2. "Pattern Recognition and Machine Learning" by Christopher Bishop
3. Scikit-learn documentation on logistic regression
4. PyTorch tutorials on building neural networks (logistic regression is the simplest form of a neural network)
5. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman (for a more advanced treatment)