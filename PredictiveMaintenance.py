"""
Project: Predictive Maintenance & RAM Pricing Intelligence System
Author: Chris D'Costa
Description: End-to-end Machine Learning pipeline handling data preprocessing,
             feature engineering, predictive maintenance classification/RUL estimation,
             and RAM market pricing regression.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def generate_mock_predictive_maintenance_data(num_samples=1000):
  """Generates synthetic multi-sensor industrial time-series data for testing."""
  np.random.seed(42)
  temperature = np.random.normal(70, 10, num_samples)
  vibration = np.random.normal(0.5, 0.1, num_samples)
  pressure = np.random.normal(30, 5, num_samples)
  rotation_speed = np.random.normal(1500, 50, num_samples)

  # Failure logic based on sensor thresholds + noise
  failure_risk = (
      (temperature > 85).astype(int) * 0.4
      + (vibration > 0.7).astype(int) * 0.4
      + np.random.binomial(1, 0.1, num_samples) * 0.2
  )
  failure_label = (failure_risk >= 0.5).astype(int)

  df = pd.DataFrame({
      'temperature': temperature,
      'vibration': vibration,
      'pressure': pressure,
      'rotation_speed': rotation_speed,
      'failure_label': failure_label,
  })
  return df


def generate_mock_ram_pricing_data(num_samples=1000):
  """Generates synthetic RAM market data for pricing regression testing."""
  np.random.seed(42)
  capacity_gb = np.choice([8, 16, 32, 64], size=num_samples)
  frequency_mhz = np.choice([3200, 3600, 4800, 5600, 6000], size=num_samples)
  brand_encoded = np.random.randint(0, 5, size=num_samples)

  # Base price formula + noise
  base_price = (
      (capacity_gb * 35)
      + (frequency_mhz * 0.08)
      + (brand_encoded * 15)
      + np.random.normal(0, 100, num_samples)
  )
  base_price = np.clip(base_price, 1200, 25000)  # Realistic INR bounds

  df = pd.DataFrame({
      'capacity_gb': capacity_gb,
      'frequency_mhz': frequency_mhz,
      'brand_encoded': brand_encoded,
      'price_inr': base_price,
  })
  return df


def run_predictive_maintenance_pipeline():
  print('--- Executing Predictive Maintenance Pipeline ---')
  df = generate_mock_predictive_maintenance_data()

  # Feature Engineering: Rolling stats
  df['temp_rolling_mean'] = df['temperature'].rolling(window=5, min_periods=1).mean()
  df['vib_rolling_std'] = df['vibration'].rolling(window=5, min_periods=1).std().fillna(0)

  X = df.drop(columns=['failure_label'])
  y = df['failure_label']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train_scaled, y_train)

  y_pred = model.predict(X_test_scaled)
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)

  print(f'Maintenance Model Accuracy: {acc * 100:.2f}%')
  print(f'Maintenance Model F1-Score: {f1:.3f}\n')


def run_ram_pricing_pipeline():
  print('--- Executing RAM Pricing Intelligence Pipeline ---')
  df = generate_mock_ram_pricing_data()

  X = df.drop(columns=['price_inr'])
  y = df['price_inr']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train_scaled, y_train)

  y_pred = model.predict(X_test_scaled)
  mae = mean_absolute_error(y_test, y_pred)
  rmse = root_mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print(f'RAM Pricing MAE: ₹{mae:.2f}')
  print(f'RAM Pricing RMSE: ₹{rmse:.2f}')
  print(f'RAM Pricing R2 Score: {r2:.3f}\n')


if __name__ == '__main__':
  print('Starting Industrial ML Pipeline Execution...\n')
  run_predictive_maintenance_pipeline()
  run_ram_pricing_pipeline()
  print('Pipeline execution completed successfully.')
