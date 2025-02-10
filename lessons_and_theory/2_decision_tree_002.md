# Lesson 2: Decision Trees for House Price Prediction

## Introduction

Decision trees are a versatile machine learning model for both classification and regression tasks. In this lesson, we'll use decision trees to predict house prices based on features like location, size, and amenities.

Imagine you're a real estate agent trying to estimate the fair price of a house based on its characteristics. This is where decision trees can help. They learn a set of rules from historical data to make predictions on new, unseen houses.

Essentially, a decision tree is used to make predictions on the target variable - say price - by recursively splitting the data based on the values of the features, choosing splits that maximize the similarity of the target variable (prices) within each subset.

The result is a tree-like model of decisions and their consequences.

By the end of this lesson, you'll understand how decision trees work, how to train and interpret them, and how they compare to other models for regression tasks.

# Table of Contents

1. [Introduction](#introduction)
2. [Intuition Behind Decision Trees](#intuition-behind-decision-trees)
3. [Anatomy of a Decision Tree](#anatomy-of-a-decision-tree)
4. [Splitting Criteria Explained](#splitting-criteria-explained)
5. [Loading and Exploring the Data](#loading-and-exploring-the-data)
6. [Data Preprocessing Deep Dive](#data-preprocessing-deep-dive)
7. [Feature Engineering](#feature-engineering)
8. [Building and Training Decision Tree Models](#building-and-training-decision-tree-models)
9. [Analyzing Feature Subset Impact](#analyzing-feature-subset-impact)
10. [Model Comparison: Linear Regression, Decision Tree, Random Forest](#model-comparison-linear-regression-decision-tree-random-forest)
11. [Bias-Variance Trade-off](#bias-variance-trade-off)
12. [Model Interpretability](#model-interpretability)
13. [Advanced Techniques](#advanced-techniques)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Ensemble Methods: Random Forests](#ensemble-methods-random-forests)
14. [Limitations of Decision Trees](#limitations-of-decision-trees)
15. [Ethical Considerations](#ethical-considerations) 
16. [Conclusion](#conclusion)
17. [Further Reading](#further-reading)


## Intuition Behind Decision Trees

Imagine you're trying to predict the price of a house based on its features. You might start by asking broad questions like "Is it in a desirable location?" and then progressively get more specific: "How many bedrooms does it have? What's the square footage?".

At each step, you're trying to split the houses into groups that are as similar as possible in terms of price. This is exactly how a decision tree works - it asks a series of questions about the features, each time trying to split the data into more homogeneous subsets.

## Anatomy of a Decision Tree

A decision tree is composed of:

- Nodes: Where a feature is tested
- Edges: The outcomes of the test
- Leaves: Terminal nodes that contain the final predictions

A simplified example of a house prices prediction decision tree might look like this:

![structure of a house prices prediction decision tree](../static/house-prices-decision-tree-and-structure.png)

The tree is built by splitting the data recursively, choosing at each step the feature and split point that results in the greatest reduction in impurity or error.

## Splitting Criteria Explained:

When building a decision tree, we need a way to determine the best feature and value to split on at each node. The goal is to create child nodes that are more "pure" or homogeneous than their parent node. The method for measuring this purity and choosing the best split differs between regression and classification tasks.

### For Regression Tasks (e.g., Predicting House Prices):

In regression problems, we're trying to predict a continuous value, like house prices. The goal is to split the data in a way that minimizes the variance of the target variable within each resulting group.

The most common metric used for regression trees is the Mean Squared Error (MSE). This is the default criterion used by scikit-learn's DecisionTreeRegressor. Let's break down how this works:

Imagine you're a real estate agent with a magical ability to instantly sort houses. Your goal? To group similar houses together as efficiently as possible. This is essentially what a decision tree does, but instead of magical powers, it uses mathematics. Let's dive in!

#### Mean Squared Error (MSE)

Imagine you're playing a house price guessing game. Your goal is to guess the prices of houses as accurately as possible.

Let's say we have 5 houses, and their actual prices are:
```
House 1: £200,000
House 2: £250,000
House 3: £180,000
House 4: £220,000
House 5: £300,000
```

#### Step 1: Calculate the average price
`(200,000 + 250,000 + 180,000 + 220,000 + 300,000) / 5 = £230,000`

So, your guess for any house would be £230,000.

#### Step 2: Calculate how wrong you are for each house
```
House 1: 230,000 - 200,000 = 30,000 
House 2: 230,000 - 250,000 = -20,000
House 3: 230,000 - 180,000 = 50,000
House 4: 230,000 - 220,000 = 10,000
House 5: 230,000 - 300,000 = -70,000
```

#### Step 3: Square these differences
```
House 1: 30,000² = 900,000,000
House 2: (-20,000)² = 400,000,000
House 3: 50,000² = 2,500,000,000
House 4: 10,000² = 100,000,000
House 5: (-70,000)² = 4,900,000,000
```
#### Step 4: Add up all these squared differences
`
900,000,000 + 400,000,000 + 2,500,000,000 + 100,000,000 + 4,900,000,000 = 8,800,000,000
`
#### Step 5: Divide by the number of houses

`8,800,000,000 ÷ 5 = 1,760,000,000`

This final number, 1,760,000,000, is your Mean Squared Error (MSE).

In mathematical notation, this whole process looks like:

$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y})^2$

Let's break this down:
- $n$ is the number of houses (5 in our example)
- $y_i$ is the actual price of each house
- $\hat{y}$ is your guess (the average price, £230,000 in our example)
- $\sum_{i=1}^n$ means "add up the following calculation for each house from the first to the last"
- The $i$ in $y_i$ is just a counter, going from 1 to $n$ (1 to 5 in our example)

As a python function, this would look like:
```
def calculate_mse(actual_prices, predicted_price):
    n = len(actual_prices)
    squared_errors = []
    
    for actual_price in actual_prices:
        error = predicted_price - actual_price
        squared_error = error ** 2
        squared_errors.append(squared_error)
    
    mse = sum(squared_errors) / n
    return mse

# Example usage
actual_prices = [200000, 250000, 180000, 220000, 300000]
predicted_price = sum(actual_prices) / len(actual_prices)  # Average price

mse = calculate_mse(actual_prices, predicted_price)
print(f"Mean Squared Error: {mse:.2f}")
```

### Evaluating Decision Points: Understanding Split Quality in Decision Trees

Now, when we split our houses into two groups, we want to measure if this split has made our predictions better. We do this by comparing the error before and after splitting using this formula:

$\Delta MSE = MSE_{before} - (({\text{fraction of houses in left group} \times MSE_{left}} + {\text{fraction of houses in right group} \times MSE_{right}}))$

Let's work through a real example to understand this:

Imagine we have 5 houses with these prices:
```
House 1: £200,000
House 2: £250,000
House 3: £180,000
House 4: £220,000
House 5: £300,000
```

We're considering splitting these houses based on whether they have more than 2 bedrooms:
- Left group (≤2 bedrooms): Houses 1, 3 (£200,000, £180,000)
- Right group (>2 bedrooms): Houses 2, 4, 5 (£250,000, £220,000, £300,000)

#### 1. First, let's calculate $MSE_{before}$
```
Mean price = (200k + 250k + 180k + 220k + 300k) ÷ 5 = £230,000

Squared differences from mean:
House 1: (230k - 200k)² = 900,000,000
House 2: (230k - 250k)² = 400,000,000
House 3: (230k - 180k)² = 2,500,000,000
House 4: (230k - 220k)² = 100,000,000
House 5: (230k - 300k)² = 4,900,000,000

MSE_before = (900M + 400M + 2,500M + 100M + 4,900M) ÷ 5
           = 1,760,000,000
```

#### 2. Now for the left group (≤2 bedrooms):
```
Mean price = (200k + 180k) ÷ 2 = £190,000

Squared differences:
House 1: (190k - 200k)² = 100,000,000
House 3: (190k - 180k)² = 100,000,000

MSE_left = (100M + 100M) ÷ 2 = 100,000,000
```

#### 3. And the right group (>2 bedrooms):
```
Mean price = (250k + 220k + 300k) ÷ 3 = £256,667

Squared differences:
House 2: (256.67k - 250k)² = 44,448,889
House 4: (256.67k - 220k)² = 1,344,448,889
House 5: (256.67k - 300k)² = 1,877,778,889

MSE_right = (44.45M + 1,344.45M + 1,877.78M) ÷ 3 = 1,088,892,222
```

#### 4. Finally, let's put it all together:
```
ΔMSE = MSE_before - ((2/5 × MSE_left) + (3/5 × MSE_right))

The second part calculates our weighted mean MSE after splitting:

- Left group has 2/5 of the houses, so we multiply its MSE by 2/5
- Right group has 3/5 of the houses, so we multiply its MSE by 3/5

This weighting ensures each house contributes equally to our final calculation.

Let's solve it:
     = 1,760,000,000 - ((2/5 × 100,000,000) + (3/5 × 1,088,892,222))
     = 1,760,000,000 - (40,000,000 + 653,335,333)
     = 1,760,000,000 - 693,335,333        # This is our weighted mean MSE after splitting
     = 1,066,664,667                      # ΔMSE: The reduction in prediction error

The ΔMSE (1,066,664,667) represents the difference between the original MSE and the weighted average MSE after splitting. This number is always non-negative due to a fundamental property of squared errors:

1. MSE is always positive (we're squaring differences from the mean)
2. When we split a group:
   - The parent uses one mean for all samples
   - Each subgroup uses its own mean, which minimises squared errors for that subgroup
   - The subgroup means must perform at least as well as the parent mean (due to minimising squared errors locally)
   - Therefore, the weighted average MSE of subgroups cannot exceed the parent MSE

Therefore:
- ΔMSE > 0 means the split has improved predictions (as in our case)
- ΔMSE = 0 means the split makes no difference
- ΔMSE < 0 is mathematically impossible
```

The larger the ΔMSE, the more effective the split is at creating subgroups with similar house prices. Our large ΔMSE of 1,066,664,667 indicates this is a very effective split.

### A simplified decision tree algorithm in python:
In practise, you'd use a library like `sklearn` to build a decision tree, but here's a simplified version in python to illustrate the concept:

```
import numpy as np
from typing import List, Dict, Any

class House:
    def __init__(self, features: Dict[str, float], price: float):
        self.features = features
        self.price = price

def find_best_split(houses: List[House], feature: str) -> tuple:
    values = sorted(set(house.features[feature] for house in houses))
    
    best_split = None
    best_delta_mse = float('-inf')

    for i in range(len(values) - 1):
        split_point = (values[i] + values[i+1]) / 2
        left = [h for h in houses if h.features[feature] < split_point]
        right = [h for h in houses if h.features[feature] >= split_point]

        if len(left) == 0 or len(right) == 0:
            continue

        mse_before = np.var([h.price for h in houses])
        mse_left = np.var([h.price for h in left])
        mse_right = np.var([h.price for h in right])

        delta_mse = mse_before - (len(left)/len(houses) * mse_left + len(right)/len(houses) * mse_right)

        if delta_mse > best_delta_mse:
            best_delta_mse = delta_mse
            best_split = split_point

    return best_split, best_delta_mse

def build_tree(houses: List[House], depth: int = 0, max_depth: int = 3) -> Dict[str, Any]:
    if depth == max_depth or len(houses) < 2:
        return {'type': 'leaf', 'value': np.mean([h.price for h in houses])}

    features = houses[0].features.keys()
    best_feature = None
    best_split = None
    best_delta_mse = float('-inf')

    for feature in features:
        split, delta_mse = find_best_split(houses, feature)
        if delta_mse > best_delta_mse:
            best_feature = feature
            best_split = split
            best_delta_mse = delta_mse

    if best_feature is None:
        return {'type': 'leaf', 'value': np.mean([h.price for h in houses])}

    left = [h for h in houses if h.features[best_feature] < best_split]
    right = [h for h in houses if h.features[best_feature] >= best_split]

    return {
        'type': 'node',
        'feature': best_feature,
        'split': best_split,
        'left': build_tree(left, depth + 1, max_depth),
        'right': build_tree(right, depth + 1, max_depth)
    }

def predict(tree: Dict[str, Any], house: House) -> float:
    if tree['type'] == 'leaf':
        return tree['value']
    
    if house.features[tree['feature']] < tree['split']:
        return predict(tree['left'], house)
    else:
        return predict(tree['right'], house)

# Example usage
houses = [
    House({'bedrooms': 2, 'area': 80, 'distance_to_tube': 15}, 200),
    House({'bedrooms': 3, 'area': 120, 'distance_to_tube': 10}, 250),
    House({'bedrooms': 2, 'area': 75, 'distance_to_tube': 20}, 180),
    House({'bedrooms': 3, 'area': 100, 'distance_to_tube': 5}, 220),
    House({'bedrooms': 4, 'area': 150, 'distance_to_tube': 2}, 300),
    House({'bedrooms': 3, 'area': 110, 'distance_to_tube': 12}, 240),
    House({'bedrooms': 2, 'area': 70, 'distance_to_tube': 25}, 190),
    House({'bedrooms': 4, 'area': 140, 'distance_to_tube': 8}, 280),
    House({'bedrooms': 3, 'area': 130, 'distance_to_tube': 6}, 260),
    House({'bedrooms': 2, 'area': 85, 'distance_to_tube': 18}, 210)
]

tree = build_tree(houses)

def print_tree(node, indent=""):
    if node['type'] == 'leaf':
        print(f"{indent}Predict price: £{node['value']:.2f}k")
    else:
        print(f"{indent}{node['feature']} < {node['split']:.2f}")
        print(f"{indent}If True:")
        print_tree(node['left'], indent + "  ")
        print(f"{indent}If False:")
        print_tree(node['right'], indent + "  ")

print_tree(tree)

# Test prediction
new_house = House({'bedrooms': 3, 'area': 105, 'distance_to_tube': 7}, 0)  # price set to 0 as it's unknown
predicted_price = predict(tree, new_house)
print(f"\nPredicted price for new house: £{predicted_price:.2f}k")

```

### Mean Squared Error (MSE) vs Mean Absolute Error (MAE)

When evaluating our decision tree's performance, we need to understand the difference between training metrics and evaluation metrics.

![mean-squared-error-mean-absolute-error](../static/mean-squared-error-mean-absolute-error.png)

Our decision tree algorithm uses MSE as the splitting criterion but measures final performance using MAE. 

Here's why we use these different metrics:

1. **Mean Squared Error (MSE)**
   ##### Calculation: (predicted house price - actual house price)². 
   
   For example, if we predict £200,000 for a house that's actually worth £150,000, the error is £50,000 and MSE is £50,000² = £2.5 billion
   
   ##### Visualisation 
   
   If we plot how wrong our house price prediction is (like £50,000 too high or -£50,000 too low) on the x-axis, and plot the squared value of this error (like £2.5 billion) on the y-axis, we get a U-shaped curve. Because MSE squares the errors, it gives more weight to data points that are further from the mean, making it a good measure of variance within groups
   
   ##### Purpose

   The decision tree uses MSE to decide where to split data because minimizing MSE is equivalent to minimizing the variance within each group, which helps find splits that create distinct groups of house prices

2. **Mean Absolute Error (MAE)**:
   ##### Calculation: |predicted house price - actual house price| 
   
   Using the same example, if we predict £200,000 for a £150,000 house, MAE is |£50,000| = £50,000
   
   ##### Visualisation 
   If we plot how wrong our prediction is on the x-axis (like £50,000 too high or -£50,000 too low), and plot the absolute value of this error on the y-axis (always positive, like £50,000), we get a V-shaped curve
   
   ##### Purpose 
   We use MAE to evaluate our final model because it's easier to understand - it directly tells us how many pounds we're off by on average

\
The decision tree uses MSE's mathematical properties to make splitting decisions, but we report MAE because "off by £50,000 on average" makes more sense than "off by £2.5 billion squared pounds"!

Here's an example to illustrate the difference:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error 
y_true = [100, 200, 300]
y_pred = [90, 210, 320]

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
```

Output:
```
Mean Squared Error: 200.00
Mean Absolute Error: 13.33
```

In this example, MSE and MAE provide different views of the error. MSE is more sensitive to the larger error (20) in the third prediction, while MAE treats all errors equally.

For house price prediction, MAE is often preferred as it directly translates to the average error in pounds. However, MSE is still commonly used as a splitting criterion in decision trees because minimizing MSE helps create groups with similar target values by minimizing the variance within each group.

### For Classification Tasks (e.g., Predicting if a House Will Sell Quickly):

In classification problems, we're trying to predict a categorical outcome, like whether a house will sell quickly or not. The goal is to split the data in a way that maximizes the "purity" of the classes within each resulting group.

There are several metrics used for classification trees, with the most common being Gini Impurity and Entropy. These metrics measure how mixed the classes are within a group.

Let's explore how different distributions of marbles affect our measures of impurity. We will then explore information gain, a measure used in conjuction with impurity metrics to decide how to split the data.

We'll use red marbles to represent quick-selling houses and blue marbles for slow-selling houses.

#### 1. Gini Impurity:
   Gini Impurity measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution in the set.

   Formula: $Gini = 1 - \sum_{i=1}^{c} (p_i)^2$

   Where $c$ is the number of classes and $p_i$ is the probability of an object being classified to a particular class.

   Let's compare three scenarios:

```
   a) 10 marbles: 7 red, 3 blue
      Fraction of red = 7/10 = 0.7
      Fraction of blue = 3/10 = 0.3
      
      Gini = 1 - (0.7² + 0.3²) = 1 - (0.49 + 0.09) = 1 - 0.58 = 0.42
```

```
   b) 10 marbles: 5 red, 5 blue
      Fraction of red = 5/10 = 0.5
      Fraction of blue = 5/10 = 0.5
      
      Gini = 1 - (0.5² + 0.5²) = 1 - (0.25 + 0.25) = 1 - 0.5 = 0.5
      most impure set
```

```
   c) 10 marbles: 9 red, 1 blue
      Fraction of red = 9/10 = 0.9
      Fraction of blue = 1/10 = 0.1
      
      Gini = 1 - (0.9² + 0.1²) = 1 - (0.81 + 0.01) = 1 - 0.82 = 0.18
      purest set
```

**The lower the Gini Impurity, the purer the set. Scenario (c) has the lowest Gini Impurity, indicating it's the most homogeneous.**

#### 2. Entropy:

Entropy is another measure of impurity, based on the concept of information theory. It quantifies the amount of uncertainty or randomness in the data.

$Entropy = -\sum_{i=1}^{c} p_i \log_2(p_i)$

Where $c$ is the number of classes and $p_i$ is the probability of an object being classified to a particular class.

Imagine you're playing a guessing game with marbles in a bag. Entropy measures how surprised you'd be when pulling out a marble. The more mixed the colours, the more surprised you might be, and the higher the entropy.

#### Let's use our marble scenarios:

10 marbles: 7 red, 3 blue

To calculate entropy, we follow these steps:

1. Calculate the fraction of each colour:
```
   Red: 7/10 = 0.7
   Blue: 3/10 = 0.3
```

2. For each colour, multiply its fraction by the log2 of its fraction:   
```
   Red: 0.7 × log2(0.7) = 0.7 × -0.5146 = -0.360
   Blue: 0.3 × log2(0.3) = 0.3 × -1.7370 = -0.5211
```

3. Sum these values and negate the result:
```
Entropy = -(-0.3602 + -0.5211) = 0.8813
```

#### Let's do this for all scenarios:

a) 7 red, 3 blue
```
   Entropy = 0.8813
```
b) 5 red, 5 blue
```
   Red: 0.5 × log2(0.5) = 0.5 × -1 = -0.5
   Blue: 0.5 × log2(0.5) = 0.5 × -1 = -0.5
   Entropy = -(-0.5 + -0.5) = 1

Highest entropy, least predictable set
```

c) 9 red, 1 blue
```
   Red: 0.9 × log2(0.9) = 0.9 × -0.1520 = -0.1368
   Blue: 0.1 × log2(0.1) = 0.1 × -3.3219 = -0.3322
   Entropy = -(-0.1368 + -0.3322) = 0.4690

Lowest entropy, most predictable set
```

Lower entropy means less surprise or uncertainty. Scenario (c) has the lowest entropy, confirming it's the most predictable (or least mixed) set.

In Python, we could calculate entropy like this:

```python
import math

def calculate_entropy(marbles):
    total = sum(marbles.values())
    entropy = 0
    for count in marbles.values():
        fraction = count / total
        entropy -= fraction * math.log2(fraction)
    return entropy

# Example usage
scenario_a = {"red": 7, "blue": 3}
entropy_a = calculate_entropy(scenario_a)
print(f"Entropy for scenario A: {entropy_a:.4f}")
```

#### 3. Information Gain:

Information Gain measures how much a split improves our ability to predict the outcome. It's a way of measuring how much better you've sorted your marbles after dividing them into groups.

Formula: $IG(T, a) = I(T) - \sum_{v \in values(a)} \frac{|T_v|}{|T|} I(T_v)$

Where:
- $T$ is the parent set
- $a$ is the attribute on which the split is made
- $v$ represents each possible value of attribute $a$
- $T_v$ is the subset of $T$ for which attribute $a$ has value $v$
- $I(T)$ is the impurity measure (Entropy or Gini) of set $T$


#### Let's use a scenario to calculate Information Gain:

We have 20 marbles total, and we're considering splitting them based on a feature (e.g., house size: small or large).
```
Before split: 12 red, 8 blue
```

Step 1: Calculate the entropy before the split
```
Entropy_before = 0.9710 (calculated as we did above)
```

After split:
```
Small houses: 8 red, 2 blue
Large houses: 4 red, 6 blue
```
Step 2: Calculate entropy for each group after the split
Entropy_small = 0.7219 (calculated for 8 red, 2 blue)
Entropy_large = 0.9710 (calculated for 4 red, 6 blue)

Step 3: Calculate the weighted average of the split entropies
```
Weight_small = 10/20 = 0.5 (half the marbles are in small houses)
Weight_large = 10/20 = 0.5 (half the marbles are in large houses)
Weighted_entropy_after = (0.5 × 0.7219) + (0.5 × 0.9710) = 0.8465
```

Step 4: Calculate Information Gain
```
Information Gain = Entropy_before - Weighted_entropy_after
                 = 0.9710 - 0.8465
                 = 0.1245
```

This positive Information Gain indicates that the split has improved our ability to predict marble colours (or in our house analogy, to predict if a house will sell quickly).

#### In Python, we could calculate Information Gain like this:

```python
def calculate_information_gain(before, after):
    entropy_before = calculate_entropy(before)
    
    total_after = sum(sum(group.values()) for group in after)
    weighted_entropy_after = sum(
        (sum(group.values()) / total_after) * calculate_entropy(group)
        for group in after
    )
    
    return entropy_before - weighted_entropy_after

# Example usage
before_split = {"red": 12, "blue": 8}
after_split = [
    {"red": 8, "blue": 2},  # Small houses
    {"red": 4, "blue": 6}   # Large houses
]

info_gain = calculate_information_gain(before_split, after_split)
print(f"Information Gain: {info_gain:.4f}")
```

#### Comparison: Splits with Different Information Gains

The decision tree algorithm always chooses the split that provides the most Information Gain. 

Let's consider two potential splits of our 20 marbles:

1. Split by house size (small vs large):
   - Small houses: 8 red, 2 blue
   - Large houses: 4 red, 6 blue
   - Information Gain: 0.1245

2. Split by garage presence:
   - Houses with garage: 6 red, 4 blue
   - Houses without garage: 6 red, 4 blue
   - Information Gain: 0

The algorithm would choose the split by house size because it provides more Information Gain. 

Zero Information Gain occurs when a split doesn't change the distribution of the target variable (in this case, marble colours or house selling speed). This happens when the proportions in each resulting group are identical to the proportions in the parent group.

In practice, splits with exactly zero Information Gain are rare. More commonly, you'll see splits with varying degrees of positive Information Gain, and the algorithm will choose the one with the highest value.

Features that provide little or no Information Gain are typically less valuable for prediction and should be considered for removal from the model. Eliminating these low-impact features can simplify the model, potentially improving its generalization ability and computational efficiency without significantly compromising predictive performance.

## Theory Conclusion

Now that we've explored the key concepts behind decision trees, let's summarize the main points and how they apply to our house price prediction task:

1. **Regression Trees vs Classification Trees**: 
   For our house price prediction problem, we're using regression trees. Unlike classification trees which use metrics like Gini impurity or entropy, regression trees aim to minimize the variance in the target variable (house prices) within each node.

2. **Splitting Criterion**: 
   In regression trees, the splitting criterion is typically the reduction in Mean Squared Error (MSE) or equivalently, the reduction in variance. At each node, the algorithm chooses the feature and split point that maximizes this reduction:

   $\Delta MSE = MSE_{parent} - (w_{left} * MSE_{left} + w_{right} * MSE_{right})$

   Where $w_{left}$ and $w_{right}$ are the proportions of samples in the left and right child nodes.

3. **Recursive Splitting**: 
   The tree is built by recursively applying this splitting process, creating a hierarchy of decision rules. This continues until a stopping condition is met, such as a maximum tree depth or a minimum number of samples per leaf.

4. **Prediction**: 
   To predict the price of a new house, we follow the decision rules from the root to a leaf node. The prediction is typically the mean price of all houses in that leaf node.

5. **Interpretability**: 
   One of the key advantages of decision trees is their interpretability. We can visualize the tree and follow its decision path, which provides insights into which features are most important for predicting house prices.

6. **Bias-Variance Trade-off**: 
   Deeper trees can capture more complex relationships in the data but risk overfitting (high variance). Shallower trees are more generalizable but might oversimplify the problem (high bias). Finding the right balance is crucial for optimal performance.

7. **Feature Importance**: 
   Decision trees naturally perform feature selection. Features that appear higher in the tree or are used in more splits are generally more important for prediction.

8. **Handling Non-linearity**: 
   Unlike linear regression, decision trees can capture non-linear relationships between features and the target variable, which is often the case in real estate markets.

9. **Limitations**: 
   Decision trees can be unstable (small changes in data can result in very different trees) and may struggle with smooth, linear relationships. These limitations are often addressed by ensemble methods like Random Forests.

As we move forward to apply these concepts to our London housing dataset, keep in mind that while the theory provides the foundation, the real insights often come from experimenting with the data, tuning the model, and interpreting the results in the context of the problem at hand.

In the next sections, we'll see how these theoretical concepts translate into practical implementation using Python and scikit-learn, and how we can use decision trees to gain insights into the London housing market.

## Practical Application
### Let's load the London housing dataset:

```python
import pandas as pd

df = pd.read_csv('London.csv')
print(df.head())
print(df.info())
```

The dataset includes the following features for each house:

- Location: Neighborhood or borough 
- House Type: Flat, terraced, semi-detached, etc.
- Bedrooms, Bathrooms, Receptions: Number of each
- Area: Floor area in square feet
- Price: Sale price 

We also have the City/County and Postal Code, which we can use to extract more geographical information.

## Data Preprocessing Deep Dive

Let's preprocess the data:

1. Convert 'Price' to numeric
2. Extract 'Postcode Area' and 'Outcode'
3. Encode categorical variables
4. Drop unnecessary columns

```python
# Convert Price to numeric
df['Price'] = df['Price'].str.replace('£', '').str.replace(',', '').astype(float)

# Extract Postcode Area and Outcode
if ' ' not in df['Postal Code'].str:
    print("Warning: Some postcodes are missing spaces.")

df['Postcode Area'] = df['Postal Code'].str.split().str[0].str.replace(r'\d', '')  
df['Postcode Outcode'] = df['Postal Code'].str.split().str[0]

# Encode categoricals
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for col in ['Location', 'House Type', 'City/County', 'Postcode Area', 'Postcode Outcode']:
    df[col] = label_encoder.fit_transform(df[col])

# Drop unnecessary columns
cols_to_drop = ['Unnamed: 0', 'Property Name', 'Postal Code']
df = df.drop(columns=cols_to_drop)
```

## Feature Engineering

Let's brainstorm some additional features we could create:

- Distance from city center
- Distance from nearest tube station
- School quality score for the area
- Crime rate for the area
- Green space percentage for the area
- Number of amenities (shops, restaurants, etc.) within a radius

These would require additional data sources, but could potentially improve our predictions.

## Building and Training Decision Tree Models

Now let's build a decision tree model:

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Prepare features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = tree.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: £{mae:.2f}")
```

The Mean Absolute Error (MAE) is the average absolute difference between the predicted and actual house prices. The lower the MAE, the better the model.

## Analyzing Feature Subset Impact

Let's see how the model performs with different subsets of features:

```python
feature_subsets = {
    'Bedrooms, Bathrooms': ['No. of Bedrooms', 'No. of Bathrooms'],
    '+ Reception': ['No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions'],
    '+ Area': ['No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'Area in sq ft'],
    '+ House Type': ['No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'Area in sq ft', 'House Type'],
    '+ Location': ['No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'Area in sq ft', 'House Type', 'Location'], 
    '+ City/County': ['No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'Area in sq ft', 'House Type', 'Location', 'City/County'],
    '+ Postcode Area': ['No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'Area in sq ft', 'House Type', 'Location', 'City/County', 'Postcode Area'],
    '+ Postcode Outcode': ['No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'Area in sq ft', 'House Type', 'Location', 'City/County', 'Postcode Area', 'Postcode Outcode'], 
}

results = {}

for subset_name, features in feature_subsets.items():
    X = df[features]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[subset_name] = mae

print(results)
```

The postcode-related features provide a significant boost in performance, likely because they capture location information at a more granular level.

<img src="feature_subset_performance.png" alt="Feature Subset Performance" width="600"/>

## Model Comparison: Linear Regression, Decision Tree, Random Forest

Let's compare our decision tree to a linear regression and a random forest:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Prepare features and target
X = df[['No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'Area in sq ft', 'House Type', 'Location', 'City/County', 'Postcode Area', 'Postcode Outcode']]
y = df['Price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f"Linear Regression MAE: £{mae_lr:.2f}")

# Decision Tree
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
print(f"Decision Tree MAE: £{mae_tree:.2f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest MAE: £{mae_rf:.2f}")
```

The random forest performs the best, followed by the decision tree, and then linear regression. This suggests that the relationship between the features and house prices is likely non-linear and complex.

The random forest's success is likely due to its ability to reduce overfitting (by averaging many trees) and capture complex interactions between features (by considering subsets of features at each split).

However, the decision tree and linear regression models are more interpretable - we can examine the learned coefficients or decision rules to understand how they make predictions.

## Bias-Variance Trade-off

The performance differences between these models illustrate the bias-variance trade-off.

- Bias is the error introduced by approximating a complex problem with a simple model. 
- Variance is the error introduced by a model's sensitivity to small fluctuations in the training data.

Linear regression has high bias (it assumes a linear relationship) but low variance. Decision trees have low bias (they can capture complex relationships) but high variance (they can overfit to noise in the training data). Random forests reduce the variance of individual trees by averaging many trees.

The goal is to find a model with low bias and low variance, but there's often a trade-off between the two.

## Model Interpretability

One of the key advantages of decision trees is their interpretability.

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=['Price'])
plt.show()
```

<img src="decision_tree_plot.png" alt="Decision Tree Plot" width="800"/>

Each node shows the feature and threshold used for splitting, the number of samples in that node, the average price in that node, and the impurity (MSE).

By following the path from the root to a leaf, we can understand the decision rules the tree has learned. For example:

"If the house is not in Wimbledon, has an area less than 2000 sq ft, and is a flat, then the predicted price is £725,000."

These rules give us insight into what factors the model considers important and how it combines them to make predictions.

## Advanced Techniques

### Hyperparameter Tuning

We can improve our model's performance by tuning its hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4]
}

tree = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best MAE: £{-grid_search.best_score_:.2f}")

best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE with best parameters: £{mae:.2f}")
```

This searches over different combinations of `max_depth`, `min_samples_split`, and `min_samples_leaf` to find the best performing model.

### Ensemble Methods: Random Forests

Random forests are an ensemble of decision trees that can achieve better performance:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"Random Forest MAE: £{mae_rf:.2f}")
```

Random forests work by:

1. Building many trees on different subsets of the data
2. Considering only a random subset of features at each split
3. Averaging the predictions of all trees

This reduces overfitting (high variance) while maintaining the ability to capture complex relationships (low bias).

## Limitations of Decision Trees

While powerful, decision trees have some limitations:

1. **Overfitting**: Deep trees can learn rules that are too specific to the training data.
2. **Instability**: Small changes in the data can result in very different trees. 
3. **Bias towards features with many levels**: Trees prefer to split on features with many distinct values.
4. **Difficulty capturing some relationships**: Trees struggle to model linear or smooth relationships.
5. **High variance**: Predictions can vary significantly based on the specific training data used.

Ensemble methods like random forests can mitigate some of these issues.

## Ethical Considerations

When using machine learning for real-world applications like house price prediction, it's important to consider the potential ethical implications:

- **Bias**: If the training data contains historical biases, the model may perpetuate these biases in its predictions.

- **Transparency**: If the model is used to make important decisions (like mortgage approvals), there may be a legal or moral obligation to explain how it makes predictions.

- **Privacy**: The model uses detailed personal information, so it's crucial to ensure that data is collected, stored, and used responsibly.

As machine learning practitioners, it's our duty to strive for models that are fair, transparent, and respectful of privacy. This may involve techniques like bias auditing, model interpretability tools, and differential privacy.

## Conclusion

In this lesson, we've covered:

- The intuition behind decision trees and how they make predictions
- Different splitting criteria, including MSE and MAE
- Preprocessing data for decision tree models, handling missing values, and feature engineering
- Training and evaluating decision trees in scikit-learn
- The impact of different feature subsets on model performance
- Comparing decision trees to linear regression and random forests
- The bias-variance trade-off and how it relates to model selection
- Interpreting decision tree models and analyzing feature importances
- Advanced techniques like hyperparameter tuning and ensemble methods
- The limitations of decision trees and ethical considerations in their use

Decision trees are a powerful and interpretable tool for regression and classification tasks. While they have limitations, they form the foundation for more advanced methods like random forests and gradient boosting.

Understanding decision trees is crucial for any machine learning practitioner. They provide a solid grounding in the core concepts of supervised learning, and their interpretability makes them invaluable for explaining predictions to stakeholders.

In the next lesson, we'll dive deeper into ensemble methods with random forests, seeing how they can improve upon the performance of single decision trees.

## Further Reading

- [Scikit-learn documentation on decision trees](https://scikit-learn.org/stable/modules/tree.html)
- [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [An Introduction to Statistical Learning, Chapter 8: Tree-Based Methods](http://faculty.marshall.usc.edu/gareth-james/ISL/)
- [Elements of Statistical Learning, Chapter 9: Additive Models, Trees, and Related Methods](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Kaggle course on Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability)
- [Google's Machine Learning Crash Course, Descending into ML: Training and Loss](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)
- [Interpretable Machine Learning, A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/)

These resources will help deepen your understanding of decision trees and their place in the broader machine learning landscape. They cover the mathematical underpinnings, practical considerations, and cutting-edge techniques in model interpretability and explainability.

Machine learning is a vast and rapidly evolving field, and there's always more to learn. I encourage you to actively experiment with these models, tune their parameters, and test them on different datasets. Hands-on experience is invaluable for building intuition and understanding.

As you progress in your machine learning journey, always keep the end goal in mind: creating models that are not only accurate, but also transparent, fair, and beneficial to society. The technical skills are important, but the ethical considerations are just as crucial.

I hope this lesson has provided a solid foundation for your exploration of decision trees and machine learning. Feel free to reach out if you have any further questions!