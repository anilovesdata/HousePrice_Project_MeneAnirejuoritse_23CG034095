# House Price Prediction - Model Development
# Dataset: House Prices: Advanced Regression Techniques
# Selected Features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("Loading dataset...")
# For this example, assuming train.csv is in the same directory
# Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
df = pd.read_csv('train.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2: FEATURE SELECTION
# ============================================================
# Select 6 features from the recommended 9
selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 
                     'GarageCars', 'YearBuilt', 'Neighborhood']
target = 'SalePrice'

# Create working dataframe
data = df[selected_features + [target]].copy()
print(f"\nSelected features: {selected_features}")
print(f"Working dataset shape: {data.shape}")

# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================
print("\n--- DATA PREPROCESSING ---")

# 3a. Check missing values
print("\nMissing values before handling:")
print(data.isnull().sum())

# 3b. Handle missing values
# Fill numeric columns with median
numeric_features = ['TotalBsmtSF', 'GarageCars']
for col in numeric_features:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

# Fill categorical with mode
if data['Neighborhood'].isnull().sum() > 0:
    data['Neighborhood'].fillna(data['Neighborhood'].mode()[0], inplace=True)

print("\nMissing values after handling:")
print(data.isnull().sum())

# 3c. Encode categorical variables
print("\n--- ENCODING CATEGORICAL VARIABLES ---")
label_encoder = LabelEncoder()
data['Neighborhood_encoded'] = label_encoder.fit_transform(data['Neighborhood'])

# Create model directory if it doesn't exist
import os
os.makedirs('model', exist_ok=True)

# Save the label encoder for later use
joblib.dump(label_encoder, 'model/neighborhood_encoder.pkl')
print(f"Neighborhoods encoded: {len(label_encoder.classes_)} unique values")

# Drop original categorical column
data = data.drop('Neighborhood', axis=1)

# 3d. Feature Scaling
print("\n--- FEATURE SCALING ---")
# Separate features and target
X = data.drop(target, axis=1)
y = data[target]

# Initialize scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'model/scaler.pkl')
print("Features scaled using StandardScaler")

# ============================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ============================================================
# STEP 5: MODEL TRAINING - RANDOM FOREST REGRESSOR
# ============================================================
print("\n--- MODEL TRAINING ---")
print("Algorithm: Random Forest Regressor")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================
# STEP 6: MODEL EVALUATION
# ============================================================
print("\n--- MODEL EVALUATION ---")

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTraining Set Metrics:")
print(f"  MAE:  ${train_mae:,.2f}")
print(f"  MSE:  ${train_mse:,.2f}")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  R²:   {train_r2:.4f}")

print("\nTesting Set Metrics:")
print(f"  MAE:  ${test_mae:,.2f}")
print(f"  MSE:  ${test_mse:,.2f}")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  R²:   {test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# ============================================================
# STEP 7: SAVE MODEL
# ============================================================
print("\n--- SAVING MODEL ---")

# Save the trained model
joblib.dump(model, 'model/house_price_model.pkl')
print("Model saved as: model/house_price_model.pkl")

# Save feature names for reference
feature_names = list(X.columns)
joblib.dump(feature_names, 'model/feature_names.pkl')
print("Feature names saved")

print("\n✓ Model development completed successfully!")
print("\nSaved files in 'model/' directory:")
print("  - house_price_model.pkl (trained model)")
print("  - scaler.pkl (feature scaler)")
print("  - neighborhood_encoder.pkl (label encoder)")
print("  - feature_names.pkl (feature reference)")

# ============================================================
# STEP 8: TEST MODEL LOADING
# ============================================================
print("\n--- TESTING MODEL PERSISTENCE ---")

# Load model
loaded_model = joblib.load('model/house_price_model.pkl')
loaded_scaler = joblib.load('model/scaler.pkl')
loaded_encoder = joblib.load('model/neighborhood_encoder.pkl')

# Test prediction
sample_data = X_test[:5]
predictions = loaded_model.predict(sample_data)

print("Model loaded successfully!")
print(f"Sample predictions: {predictions[:3]}")
print("\n✓ All components saved and tested successfully!")