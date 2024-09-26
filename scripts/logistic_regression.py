from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import time

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