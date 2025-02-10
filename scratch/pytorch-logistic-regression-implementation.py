def prepare_data(df: pd.DataFrame) -> Tuple[NDArray, NDArray, NDArray, NDArray, StandardScaler]:
    """Prepare data for PyTorch model training by splitting and scaling.
    
    This function follows the same preprocessing steps from our numpy implementation
    in Lesson 1A, but prepares data specifically for PyTorch:
    1. Separates features and target
    2. Creates stratified train/test split
    3. Standardizes features using training data statistics
    
    Args:
        df: DataFrame containing cancer measurements and diagnosis
            Features should be numeric measurements (e.g., cell size, shape)
            Target should be binary (0=benign, 1=malignant)
    
    Returns:
        Tuple containing:
        - X_train_scaled: Standardized training features
        - X_test_scaled: Standardized test features
        - y_train: Training labels
        - y_test: Test labels
        - scaler: Fitted StandardScaler for future use
    """
    # Separate features and target
    X = df.drop('target', axis=1).values  # Features as numpy array
    y = df['target'].values               # Labels as numpy array

    # Split data - using same 80/20 ratio and stratification as Lesson 1A
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,           # 20% test set
        random_state=42,         # For reproducibility
        stratify=y               # Maintain class balance
    )
    
    # Scale features using training data statistics
    # Note: We standardize error terms without normalizing distribution
    # because their skewness might indicate malignancy
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

class CancerDataset(Dataset):
    """PyTorch Dataset wrapper for cancer data.
    
    This class bridges our numpy arrays from prepare_data() to PyTorch's
    efficient data loading system. It:
    1. Converts numpy arrays to PyTorch tensors
    2. Provides length information for batch creation
    3. Enables indexed access for efficient mini-batch sampling
    
    Args:
        X: Feature array (standardized measurements)
        y: Label array (0=benign, 1=malignant)
    """
    def __init__(self, X: NDArray, y: NDArray):
        # Convert numpy arrays to PyTorch tensors with appropriate types
        self.X = torch.FloatTensor(X)            # Features as 32-bit float
        self.y = torch.FloatTensor(y).reshape(-1, 1)  # Labels as 2D tensor
        
    def __len__(self) -> int:
        """Return dataset size for batch calculations."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enable indexing for batch sampling."""
        return self.X[idx], self.y[idx]

class CancerClassifier(nn.Module):
    """PyTorch binary classifier for cancer diagnosis.
    
    This implements the same logistic regression model from Lesson 1A, but using
    PyTorch's neural network framework. Key components:
    1. Linear layer: Computes weighted sum (z = wx + b)
    2. Sigmoid activation: Converts sum to probability
    3. Xavier initialization: For stable training with standardized features
    
    Args:
        input_features: Number of measurements used for diagnosis
    """
    def __init__(self, input_features: int):
        super().__init__()
        # Single linear layer - matches our numpy implementation
        self.linear = nn.Linear(input_features, 1)
        # Sigmoid activation - same as Lesson 1A
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights using Xavier/Glorot initialization
        # This ensures good starting point with standardized features
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diagnosis probability.
        
        This exactly mirrors our numpy implementation:
        1. Linear combination of features
        2. Sigmoid activation for probability
        """
        return self.sigmoid(self.linear(x))
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convert probabilities to binary predictions.
        
        Args:
            x: Input features as tensor
            
        Returns:
            Binary predictions (0=benign, 1=malignant)
        """
        with torch.no_grad():  # No gradient tracking needed
            probabilities = self(x)
            # Default threshold of 0.5 - same as Lesson 1A
            return (probabilities > 0.5).float()

def train_model(
    model: CancerClassifier, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    epochs: int = 1000,
    lr: float = 0.001,
    patience: int = 5
) -> Tuple[CancerClassifier, Dict]:
    """Train cancer classifier with early stopping.
    
    This implements the same training process as Lesson 1A but with PyTorch's:
    1. Automatic differentiation for gradients
    2. Mini-batch processing for efficiency
    3. Adam optimizer for adaptive learning rates
    4. Early stopping to prevent overfitting
    
    Args:
        model: PyTorch cancer classifier
        train_loader: DataLoader for training batches
        val_loader: DataLoader for validation batches
        epochs: Maximum training iterations
        lr: Learning rate for optimization
        patience: Epochs to wait before early stopping
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Binary Cross Entropy - same loss as Lesson 1A
    criterion = nn.BCELoss()
    # Adam optimizer - handles feature scale differences well
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    # Training history for visualization
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()  # Enable gradient tracking
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass - get diagnosis probabilities
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass - learn feature importance
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights
            
            # Track metrics
            train_losses.append(loss.item())
            train_correct += ((y_pred > 0.5) == y_batch).sum().item()
            train_total += len(y_batch)
        
        # Validation phase
        model.eval()  # Disable gradient tracking
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_losses.append(criterion(y_pred, y_batch).item())
                val_correct += ((y_pred > 0.5) == y_batch).sum().item()
                val_total += len(y_batch)
        
        # Calculate epoch metrics
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}\n')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Restore best weights
    model.load_state_dict(best_weights)
    return model, history

def plot_training_curves(history: Dict[str, List[float]]) -> None:
    """Visualize training progression.
    
    Creates side-by-side plots of:
    1. Loss curves - Shows learning progression
    2. Accuracy curves - Shows diagnostic performance
    
    Args:
        history: Dict containing training metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross Entropy')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Load and prepare data
df = load_cancer_data()
X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)

# Create data loaders with reasonable batch size for medical data
batch_size = 32  # Small enough for precise updates, large enough for efficiency
train_dataset = CancerDataset(X_train_scaled, y_train)
val_dataset = CancerDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize and train model
model = CancerClassifier(input_features=X_train_scaled.shape[1])
model, history = train_model(model, train_loader, val_loader)

# Plot training results to understand learning process
plot_training_curves(history)

# Print final metrics
with torch.no_grad():
    train_preds = model(torch.FloatTensor(X_train_scaled))
    test_preds = model(torch.FloatTensor(X_test_scaled))
    
    train_acc = ((train_preds > 0.5).float().numpy().flatten() == y_train).mean()
    test_acc = ((test_preds > 0.5).float().numpy().flatten() == y_test).mean()
    
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Testing Accuracy: {test_acc:.4f}")