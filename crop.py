# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("crop_data.csv")

# Split data
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# User input
print("Enter the following details:")

N = float(input("Nitrogen: "))
P = float(input("Phosphorus: "))
K = float(input("Potassium: "))
temp = float(input("Temperature: "))
humidity = float(input("Humidity: "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall: "))

# Prediction
prediction = model.predict([[N, P, K, temp, humidity, ph, rainfall]])

print("\n🌱 Recommended Crop:", prediction[0])