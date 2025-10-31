# 🚦 Road Accident Severity Predictor using Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 1️⃣ Sample Dataset: Simulated accident records
data = pd.DataFrame({
    'Speed': [60, 45, 80, 30, 55, 70, 40, 65],
    'Weather': [1, 0, 1, 0, 1, 0, 0, 1],       # 1 = Rainy, 0 = Clear
    'RoadType': [1, 2, 1, 3, 2, 1, 3, 2],      # 1 = Highway, 2 = Urban, 3 = Rural
    'DriverAge': [25, 40, 22, 35, 30, 28, 45, 33],
    'Seatbelt': [1, 0, 1, 1, 0, 1, 0, 1],      # 1 = Worn, 0 = Not worn
    'Severity': [3, 2, 5, 1, 4, 3, 2, 4]       # Severity scale: 1 (low) to 5 (high)
})

# 2️⃣ Feature Selection
X = data[['Speed', 'Weather', 'RoadType', 'DriverAge', 'Seatbelt']]
y = data['Severity']

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4️⃣ Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5️⃣ Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"🔍 Mean Squared Error on Test Set: {mse:.2f}")

# 6️⃣ Save the Model
joblib.dump(model, 'accident_severity_model.pkl')
print("✅ Model saved as 'accident_severity_model.pkl'")

# 7️⃣ Predicting Severity for a Hypothetical Case
# Example: Speed=72, Rainy=1, RoadType=2 (Urban), DriverAge=29, Seatbelt=1
new_case = pd.DataFrame([[72, 1, 2, 29, 1]], columns=X.columns)
predicted_severity = model.predict(new_case)
print(f"🚨 Predicted Accident Severity: {predicted_severity[0]:.2f}")