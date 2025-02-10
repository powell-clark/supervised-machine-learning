from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# Convert data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegressionModel(input_dim=X_train.shape[1])

# Define loss function and optimizer  
criterion = nn.BCELoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train model
num_epochs = 1000

start_time = time.time()
for epoch in range(num_epochs):
    y_predicted = model(X_train_tensor)
    loss = criterion(y_predicted, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

end_time = time.time()
print(f"Training time: {end_time-start_time:.2f} seconds")

# Evaluate model
with torch.no_grad():
    y_predicted = model(X_test_tensor)
    y_predicted_cls = y_predicted.round()

accuracy = accuracy_score(y_test, y_predicted_cls)
precision = precision_score(y_test, y_predicted_cls)
recall = recall_score(y_test, y_predicted_cls)
f1 = f1_score(y_test, y_predicted_cls)
cm = confusion_matrix(y_test, y_predicted_cls)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}") 
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Advanced Concepts and Model Improvement

# 1. Hyperparameter Tuning
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

# 2. Feature Importance
feature_importance = abs(model.linear.weight.detach().numpy()[0])
feature_names = data.feature_names

# Sort features by importance
sorted_idx = np.argsort(feature_importance)
sorted_importance = feature_importance[sorted_idx]
sorted_names = np.array(feature_names)[sorted_idx]

# Print top 10 features (now in descending order)
for importance, name in zip(sorted_importance[::-1][:10], sorted_names[::-1][:10]):
    print(f"{name}: {importance:.4f}")

plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_importance)), sorted_importance[::-1])
plt.xticks(range(len(sorted_importance)), sorted_names[::-1], rotation=90)
plt.title('Feature Importance (Most to Least Important)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# 3. Cross-validation
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

# 4. Comparison to Baseline
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_accuracy = dummy_clf.score(X_test, y_test)

print(f"Dummy Classifier Accuracy: {dummy_accuracy:.4f}")
print(f"Our Model Accuracy: {accuracy:.4f}")