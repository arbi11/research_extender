"""
C-Core Electromagnet Force Prediction using Neural Networks

Trains a feedforward neural network to predict electromagnetic force on a
C-core armature from design parameters (coil current, gap, core width, turns).

Dataset: 500 FEMM simulations from c_core_force_dataset.csv
Architecture: [4] → [32] → [64] → [32] → [1]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("C-CORE ELECTROMAGNET FORCE PREDICTION")
print("Neural Network Training Script")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1/6] Loading dataset...")
dataset_path = Path('c_core_force_dataset.csv')
df = pd.read_csv(dataset_path)

print(f"  ✓ Loaded {len(df)} samples from {dataset_path.name}")
print(f"  Columns: {list(df.columns)}")

# Extract features and target
input_features = ['coil_current_a', 'armature_gap_mm', 'core_width_mm', 'coil_turns']
output_target = 'force_magnitude_n'

X = df[input_features].values
y = df[output_target].values

print(f"\n  Input features (X): {X.shape}")
for i, feat in enumerate(input_features):
    print(f"    {feat:20s}: [{X[:, i].min():7.2f}, {X[:, i].max():7.2f}]")

print(f"\n  Output target (y): {y.shape}")
print(f"    {output_target:20s}: [{y.min():7.4f}, {y.max():7.4f}] N")
print(f"    Mean force: {y.mean():.4f} N, Std: {y.std():.4f} N")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n[2/6] Preprocessing data...")

# Train/val/test split: 70%/20%/10%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42
)

print(f"  Dataset split:")
print(f"    Training:   {len(X_train):3d} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"    Validation: {len(X_val):3d} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"    Test:       {len(X_test):3d} samples ({len(X_test)/len(X)*100:.1f}%)")

# Normalization (zero mean, unit variance)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_norm = scaler_X.fit_transform(X_train)
X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)

y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"  ✓ Features normalized (mean≈0, std≈1)")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_norm)
y_train_tensor = torch.FloatTensor(y_train_norm).unsqueeze(1)
X_val_tensor = torch.FloatTensor(X_val_norm)
y_val_tensor = torch.FloatTensor(y_val_norm).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_norm)
y_test_tensor = torch.FloatTensor(y_test_norm).unsqueeze(1)

print(f"  ✓ Converted to PyTorch tensors")

# ============================================================================
# 3. MODEL DEFINITION
# ============================================================================

print("\n[3/6] Building neural network...")

class CCoreForceANN(nn.Module):
    """
    Feedforward neural network for C-core force prediction.
    Architecture: [4] → [32] → [64] → [32] → [1]
    """
    def __init__(self, input_size=4, hidden_sizes=[32, 64, 32], output_size=1):
        super(CCoreForceANN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = CCoreForceANN(input_size=4, hidden_sizes=[32, 64, 32], output_size=1)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Architecture: [4] → [32] → [64] → [32] → [1]")
print(f"  Total parameters: {total_params:,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"  Loss function: MSE")
print(f"  Optimizer: Adam (lr=0.001)")

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

print("\n[4/6] Training neural network...")

num_epochs = 200
batch_size = 32

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_losses = []
val_losses = []

for epoch in tqdm(range(num_epochs), desc="  Training"):
    # Training phase
    model.train()
    epoch_train_loss = 0.0

    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * batch_X.size(0)

    epoch_train_loss /= len(X_train_tensor)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, y_val_tensor).item()
        val_losses.append(val_loss)

print(f"\n  ✓ Training complete!")
print(f"    Final train loss: {train_losses[-1]:.6f}")
print(f"    Final val loss:   {val_losses[-1]:.6f}")

# Check for overfitting
if val_losses[-1] > train_losses[-1] * 1.5:
    print(f"  ⚠ Warning: Validation loss significantly higher than training loss")
else:
    print(f"  ✓ No significant overfitting detected")

# Save model
torch.save(model.state_dict(), 'c_core_force_model.pth')
print(f"\n  ✓ Model saved to c_core_force_model.pth")

# ============================================================================
# 5. PERFORMANCE EVALUATION
# ============================================================================

print("\n[5/6] Evaluating model performance...")

model.eval()
with torch.no_grad():
    y_test_pred_norm = model(X_test_tensor).numpy().flatten()

# Denormalize predictions
y_test_pred = scaler_y.inverse_transform(y_test_pred_norm.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

residuals = y_test_pred - y_test

print(f"\n  Test Set Performance:")
print(f"  {'='*60}")
print(f"    MSE (Mean Squared Error):    {mse:.4f} N²")
print(f"    RMSE (Root MSE):             {rmse:.4f} N")
print(f"    MAE (Mean Absolute Error):   {mae:.4f} N")
print(f"    R² Score:                    {r2:.4f}")
print(f"    MAPE (Mean Abs % Error):     {mape:.2f}%")
print(f"  {'='*60}")

# ============================================================================
# 6. FEATURE IMPORTANCE & VISUALIZATION
# ============================================================================

print("\n[6/6] Generating visualizations...")

# Feature importance (correlation analysis)
correlations = []
for i, feature_name in enumerate(input_features):
    corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
    correlations.append((feature_name, corr))

correlations_sorted = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

print(f"\n  Feature Importance (Correlation with Force):")
for feature_name, corr in correlations_sorted:
    print(f"    {feature_name:20s}: {corr:+.4f}")

most_important_feature = correlations_sorted[0][0]
feature_idx = input_features.index(most_important_feature)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Training history
ax1 = plt.subplot(3, 2, 1)
ax1.plot(train_losses, label='Training Loss', linewidth=2)
ax1.plot(val_losses, label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss (Normalized)')
ax1.set_title('Training History')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Predicted vs True
ax2 = plt.subplot(3, 2, 2)
ax2.scatter(y_test, y_test_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('True Force [N]')
ax2.set_ylabel('Predicted Force [N]')
ax2.set_title(f'Predicted vs True Force (R² = {r2:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residual analysis
ax3 = plt.subplot(3, 2, 3)
ax3.scatter(y_test, residuals, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('True Force [N]')
ax3.set_ylabel('Residual (Predicted - True) [N]')
ax3.set_title('Residual Analysis')
ax3.grid(True, alpha=0.3)

# Plot 4: Error distribution
ax4 = plt.subplot(3, 2, 4)
ax4.hist(residuals, bins=20, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax4.set_xlabel('Prediction Error [N]')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Error Distribution (μ={residuals.mean():.3f}, σ={residuals.std():.3f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Feature importance
ax5 = plt.subplot(3, 2, 5)
feature_names_short = [f.replace('_', ' ').title() for f, _ in correlations_sorted]
correlations_abs = [abs(c) for _, c in correlations_sorted]
ax5.barh(feature_names_short, correlations_abs, color='steelblue', edgecolor='black')
ax5.set_xlabel('|Correlation with Force|')
ax5.set_title('Feature Importance')
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Sensitivity analysis (most important feature)
ax6 = plt.subplot(3, 2, 6)

# Create test range
feature_min = X_train[:, feature_idx].min()
feature_max = X_train[:, feature_idx].max()
test_range = np.linspace(feature_min, feature_max, 100)

# Fix other features at median
X_sensitivity = np.tile(np.median(X_train, axis=0), (100, 1))
X_sensitivity[:, feature_idx] = test_range

# Predict
X_sensitivity_norm = scaler_X.transform(X_sensitivity)
X_sensitivity_tensor = torch.FloatTensor(X_sensitivity_norm)

model.eval()
with torch.no_grad():
    y_sensitivity_norm = model(X_sensitivity_tensor).numpy().flatten()
    y_sensitivity = scaler_y.inverse_transform(y_sensitivity_norm.reshape(-1, 1)).flatten()

ax6.plot(test_range, y_sensitivity, linewidth=3, color='blue', label='ANN Prediction')
ax6.scatter(X_train[:, feature_idx], y_train, alpha=0.1, s=5, color='gray', label='Training Data')
ax6.set_xlabel(most_important_feature.replace('_', ' ').title())
ax6.set_ylabel('Predicted Force [N]')
ax6.set_title(f'Sensitivity: Force vs {most_important_feature.replace("_", " ").title()}')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle('C-Core Force Prediction: Neural Network Performance',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('performance_plots.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved performance_plots.png")

plt.show()

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"Model:  c_core_force_model.pth")
print(f"Plots:  performance_plots.png")
print(f"\nTest Performance: R² = {r2:.4f}, MAE = {mae:.4f} N ({mape:.2f}%)")
print("=" * 80)
