import numpy as np
import matplotlib.pyplot as plt

# Generate data
errors = np.linspace(-100, 100, 1000)  # Error values from -100 to 100
mse = errors ** 2  # Squared errors
mae = np.abs(errors)  # Absolute errors

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot MSE
ax1.plot(errors, mse, 'b-', linewidth=2)
ax1.set_title('Mean Squared Error (MSE)', fontsize=12)
ax1.set_xlabel('Prediction Error (£000s)', fontsize=10)
ax1.set_ylabel('Squared Error (£000s²)', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot MAE
ax2.plot(errors, mae, 'r-', linewidth=2)
ax2.set_title('Mean Absolute Error (MAE)', fontsize=12)
ax2.set_xlabel('Prediction Error (£000s)', fontsize=10)
ax2.set_ylabel('Absolute Error (£000s)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Add a point to highlight the non-differentiable point in MAE
ax2.plot([0], [0], 'ko', markersize=10)

plt.tight_layout()
plt.show()