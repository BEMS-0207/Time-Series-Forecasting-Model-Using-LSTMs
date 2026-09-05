"""
LSTM Power Outage Duration Prediction System
With Multiple Data Loading Options
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
import urllib.request
import zipfile
import io

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("LSTM POWER OUTAGE DURATION PREDICTION SYSTEM")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATASET WITH MULTIPLE OPTIONS
# ============================================================================

def load_dataset():
    """Try multiple sources to load the dataset"""
    
    print("\n[1/8] Loading dataset...")
    
    # Option 1: Check if file exists locally
    local_files = ['Outage_Data.xlsx', 'outage_data.xlsx', 'data.xlsx']
    for file in local_files:
        if os.path.exists(file):
            try:
                df = pd.read_excel(file)
                print(f"✓ Dataset loaded from local file: {file}")
                print(f"  - Records: {len(df)}")
                print(f"  - Columns: {len(df.columns)}")
                return df
            except:
                continue
    
    # Option 2: Try to download from Purdue
    urls = [
        "https://engineering.purdue.edu/LASCI/research-data/outages/outagerisks/Outage_Data.xlsx",
        "https://raw.githubusercontent.com/your-repo/outage-data/main/Outage_Data.xlsx",
    ]
    
    for url in urls:
        try:
            print(f"Attempting to download from: {url}")
            df = pd.read_excel(url)
            print(f"✓ Dataset downloaded successfully")
            print(f"  - Records: {len(df)}")
            print(f"  - Columns: {len(df.columns)}")
            return df
        except:
            continue
    
    # Option 3: If all fails, create sample data for demonstration
    print("\n⚠️ Could not download dataset. Creating sample data for demonstration...")
    print("NOTE: For real results, please download the dataset manually from:")
    print("https://engineering.purdue.edu/LASCI/research-data/outages/outagerisks/Outage_Data.xlsx")
    
    # Create sample data with realistic patterns
    np.random.seed(42)
    n_samples = 1534
    
    # Generate dates from 2000 to 2016
    dates = pd.date_range('2000-01-01', '2016-07-01', periods=n_samples)
    
    # Generate realistic outage data
    data = {
        'YEAR': dates.year,
        'MONTH': dates.month,
        'DAY': dates.day,
        'OUTAGE.DURATION': np.random.exponential(200, n_samples) + 50,  # Exponential distribution
        'CUSTOMERS.AFFECTED': np.random.exponential(10000, n_samples) + 1000,
        'DEMAND.LOSS': np.random.exponential(500, n_samples) + 100,
    }
    
    # Add some seasonal patterns
    data['OUTAGE.DURATION'] += 100 * np.sin(2 * np.pi * dates.month / 12)
    data['CUSTOMERS.AFFECTED'] += 5000 * np.sin(2 * np.pi * dates.month / 12)
    
    # Add some random features
    for i in range(10):
        data[f'FEATURE_{i+1}'] = np.random.randn(n_samples) * 100 + 500
    
    df = pd.DataFrame(data)
    print(f"✓ Created sample dataset with {len(df)} records")
    return df

# Load the dataset
df = load_dataset()

# ============================================================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ============================================================================

print("\n[2/8] Preprocessing data...")

# Identify target column
target_col = 'OUTAGE.DURATION'
if target_col not in df.columns:
    # Try alternative names
    alternatives = ['OUTAGE DURATION', 'DURATION', 'Outage Duration', 'duration']
    for alt in alternatives:
        if alt in df.columns:
            target_col = alt
            break

print(f"Target variable: {target_col}")

# Clean data
data = df.copy()
initial_records = len(data)

# Remove rows with missing target
data = data.dropna(subset=[target_col])
print(f"  - Removed {initial_records - len(data)} rows with missing target")

# Remove rows with zero or negative duration
data = data[data[target_col] > 0]
print(f"  - Removed rows with zero/negative duration")

# Select numeric features
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = [target_col, 'YEAR', 'MONTH', 'DAY', 'OBSERVATION.START', 'OBSERVATION.END']
features = [col for col in numeric_cols if col not in exclude_cols]

print(f"  - Selected {len(features)} numeric features")

# Handle missing values in features
for col in features:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())
        print(f"  - Filled missing values in {col}")

# Prepare X and y
X = data[features].values
y = data[target_col].values

print(f"✓ Final dataset: {len(y)} records")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[3/8] Exploratory Data Analysis...")

print(f"\nTarget Variable Statistics:")
print(f"  - Mean: {np.mean(y):.2f} minutes")
print(f"  - Median: {np.median(y):.2f} minutes")
print(f"  - Std Dev: {np.std(y):.2f} minutes")
print(f"  - Min: {np.min(y):.2f} minutes")
print(f"  - Max: {np.max(y):.2f} minutes")

# Distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(y, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(np.mean(y), color='red', linestyle='--', label=f'Mean: {np.mean(y):.0f} min')
axes[0].axvline(np.median(y), color='green', linestyle='--', label=f'Median: {np.median(y):.0f} min')
axes[0].set_xlabel('Outage Duration (minutes)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Outage Durations')
axes[0].legend()

# Box plot
axes[1].boxplot(y)
axes[1].set_ylabel('Outage Duration (minutes)')
axes[1].set_title('Box Plot of Outage Durations')

plt.tight_layout()
plt.savefig('1_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved distribution plot as '1_distribution_analysis.png'")

# ============================================================================
# STEP 4: SCALE DATA
# ============================================================================

print("\n[4/8] Scaling data...")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"✓ Data scaled to [0, 1] range")

# ============================================================================
# STEP 5: CREATE SEQUENCES
# ============================================================================

print("\n[5/8] Creating sequences...")

# Determine sequence length
seq_length = min(30, max(5, int(len(y) * 0.05)))
print(f"  - Sequence length: {seq_length}")

X_seq, y_seq = [], []
for i in range(seq_length, len(X_scaled)):
    X_seq.append(X_scaled[i-seq_length:i])
    y_seq.append(y_scaled[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"✓ Created {len(X_seq)} sequences")

# ============================================================================
# STEP 6: CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================================

print("\n[6/8] Splitting data chronologically...")

split_idx = int(len(X_seq) * 0.8)
X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
X_test, y_test = X_seq[split_idx:], y_seq[split_idx:]

print(f"  - Training: {len(X_train)} samples ({len(X_train)/len(X_seq)*100:.1f}%)")
print(f"  - Testing: {len(X_test)} samples ({len(X_test)/len(X_seq)*100:.1f}%)")

# ============================================================================
# STEP 7: BUILD AND TRAIN LSTM MODEL
# ============================================================================

print("\n[7/8] Building and training LSTM model...")

model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, 
         input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nModel Architecture:")
model.summary()

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print(f"✓ Model trained for {len(history.history['loss'])} epochs")

# ============================================================================
# STEP 8: EVALUATE MODEL
# ============================================================================

print("\n[8/8] Evaluating model...")

# Make predictions
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)
print(f"\n  MAE  (Mean Absolute Error):  {mae:.2f} minutes")
print(f"  MSE  (Mean Squared Error):   {mse:.2f} minutes²")
print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} minutes")
print(f"  R²   (Coefficient of Determination): {r2:.4f}")
print("=" * 60)

# ============================================================================
# STEP 9: VISUALIZE RESULTS
# ============================================================================

print("\nGenerating visualizations...")

# Figure 1: Training and Validation Loss
fig1, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Training and Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved training loss plot as '2_training_loss.png'")

# Figure 2: Actual vs Predicted
fig2, ax = plt.subplots(figsize=(14, 6))
x_axis = range(len(y_actual))
ax.plot(x_axis, y_actual.flatten(), 'b-', label='Actual Duration', linewidth=2, alpha=0.8)
ax.plot(x_axis, y_pred.flatten(), 'r-', label='Predicted Duration', linewidth=2, alpha=0.8)
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Outage Duration (minutes)')
ax.set_title('Actual vs Predicted Outage Duration (Test Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved actual vs predicted plot as '3_actual_vs_predicted.png'")

# Figure 3: Scatter Plot
fig3, ax = plt.subplots(figsize=(8, 8))
min_val = min(np.min(y_actual), np.min(y_pred))
max_val = max(np.max(y_actual), np.max(y_pred))
ax.scatter(y_actual, y_pred, alpha=0.5, s=30)
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Outage Duration (minutes)')
ax.set_ylabel('Predicted Outage Duration (minutes)')
ax.set_title(f'Predicted vs Actual (R² = {r2:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved scatter plot as '4_scatter_plot.png'")

# Figure 4: Residual Plot
fig4, ax = plt.subplots(figsize=(12, 5))
residuals = y_actual.flatten() - y_pred.flatten()
ax.scatter(y_pred.flatten(), residuals, alpha=0.5, s=30)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Duration (minutes)')
ax.set_ylabel('Residual (Actual - Predicted)')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('5_residual_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved residual plot as '5_residual_plot.png'")

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================

print("\nSaving results...")

# Save metrics to file
with open('model_metrics.txt', 'w') as f:
    f.write("LSTM POWER OUTAGE DURATION PREDICTION - MODEL METRICS\n")
    f.write("=" * 60 + "\n")
    f.write(f"MAE:  {mae:.4f} minutes\n")
    f.write(f"MSE:  {mse:.4f} minutes²\n")
    f.write(f"RMSE: {rmse:.4f} minutes\n")
    f.write(f"R²:   {r2:.4f}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("MODEL CONFIGURATION\n")
    f.write("=" * 60 + "\n")
    f.write(f"Total Records: {len(y)}\n")
    f.write(f"Features Used: {len(features)}\n")
    f.write(f"Sequence Length: {seq_length}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Testing Samples: {len(X_test)}\n")
    f.write(f"Epochs Trained: {len(history.history['loss'])}\n")
    f.write(f"Batch Size: 32\n")
    f.write(f"LSTM Units: 64, 32\n")
    f.write(f"Dropout: 0.2\n")
    f.write(f"Optimizer: Adam\n")
    f.write(f"Loss Function: MSE\n")
    f.write(f"Features: {features}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("NOTE: Using sample data (real dataset could not be loaded)\n")

print("✓ Saved metrics to 'model_metrics.txt'")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_actual.flatten(),
    'Predicted': y_pred.flatten(),
    'Residual': residuals.flatten()
})
predictions_df.to_csv('predictions.csv', index=False)
print("✓ Saved predictions to 'predictions.csv'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. 1_distribution_analysis.png - Outage duration distribution")
print("  2. 2_training_loss.png - Training and validation loss")
print("  3. 3_actual_vs_predicted.png - Actual vs predicted line plot")
print("  4. 4_scatter_plot.png - Scatter plot with R²")
print("  5. 5_residual_plot.png - Residual analysis")
print("  6. model_metrics.txt - All evaluation metrics")
print("  7. predictions.csv - Actual and predicted values")
print("\n" + "=" * 80)

if "sample" in str(df).lower():
    print("\n⚠️ NOTE: Using SAMPLE DATA. For real results:")
    print("   1. Download dataset from: https://engineering.purdue.edu/LASCI/research-data/outages/outagerisks/Outage_Data.xlsx")
    print("   2. Save as 'Outage_Data.xlsx' in the current directory")
    print("   3. Run the script again")
