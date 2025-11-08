import pandas as pd
from diabetes_model_files.custom_model import CustomModel

print("--- 1. Initializing CustomModel ---")
# We must provide the correct path to the data, just like app.py does
model = CustomModel(data='diabetesbal.csv')

print("\n--- 2. Training Model ---")
# This will train the model and set the all-important self.feature_columns
model.train_model()

print("\n--- 3. Preparing Test Data (Diabetic Case) ---")
# This is a sample of 'completed_data' that the RAG pipeline would provide
test_data_1 = {
    'age': 55.0,
    'gender': 'Male',
    'bmi': 30.1,
    'hypertension': 1,
    'heart_disease': 0,
    'blood_glucose_level': 180.0
}

print(f"Test Input: {test_data_1}")
prediction_1 = model.predict(test_data_1)
print(f"Prediction: {prediction_1}")


print("\n--- 4. Preparing Test Data (Non-Diabetic Case) ---")
test_data_2 = {
    'age': 25.0,
    'gender': 'Female',
    'bmi': 22.0,
    'hypertension': 0,
    'heart_disease': 0,
    'blood_glucose_level': 90.0
}

print(f"Test Input: {test_data_2}")
prediction_2 = model.predict(test_data_2)
print(f"Prediction: {prediction_2}")