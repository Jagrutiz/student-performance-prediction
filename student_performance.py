import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- Generate Sample Dataset ---
np.random.seed(42)
n_samples = 500

data = {
    'study_hours': np.random.uniform(1, 10, n_samples),
    'attendance_pct': np.random.uniform(50, 100, n_samples),
    'prev_grade': np.random.uniform(40, 100, n_samples),
    'assignments_done': np.random.randint(0, 10, n_samples),
    'sleep_hours': np.random.uniform(4, 9, n_samples),
}

df = pd.DataFrame(data)

# Target: final score (weighted combination + noise)
df['final_score'] = (
    0.35 * df['study_hours'] * 7 +
    0.25 * df['attendance_pct'] * 0.6 +
    0.20 * df['prev_grade'] * 0.5 +
    0.10 * df['assignments_done'] * 5 +
    0.10 * df['sleep_hours'] * 5 +
    np.random.normal(0, 3, n_samples)
).clip(0, 100)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData statistics:")
print(df.describe().round(2))

# --- Preprocessing ---
X = df.drop('final_score', axis=1)
y = df['final_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: Linear Regression ---
print("\n--- Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

lr_r2 = r2_score(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
print(f"R2 Score: {lr_r2:.4f}")
print(f"RMSE: {lr_rmse:.4f}")
print(f"Accuracy (within 10 marks): {np.mean(np.abs(lr_preds - y_test) < 10) * 100:.2f}%")

# --- Model 2: Neural Network ---
print("\n--- Neural Network (MLP) ---")
nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)
nn_preds = nn_model.predict(X_test_scaled)

nn_r2 = r2_score(y_test, nn_preds)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
print(f"R2 Score: {nn_r2:.4f}")
print(f"RMSE: {nn_rmse:.4f}")
print(f"Accuracy (within 10 marks): {np.mean(np.abs(nn_preds - y_test) < 10) * 100:.2f}%")

# --- Predict for a new student ---
print("\n--- Sample Prediction ---")
new_student = pd.DataFrame([{
    'study_hours': 6.5,
    'attendance_pct': 85,
    'prev_grade': 72,
    'assignments_done': 8,
    'sleep_hours': 7
}])
new_scaled = scaler.transform(new_student)
predicted_score = nn_model.predict(new_scaled)[0]
print(f"Predicted final score for new student: {predicted_score:.2f}")
print("\nDone! Model training and evaluation complete.")
