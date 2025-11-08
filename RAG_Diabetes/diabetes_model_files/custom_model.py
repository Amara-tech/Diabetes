from .helpers import DiabetesHelper
from lightgbm import LGBMClassifier
import pandas as pd
import os

class CustomModel:
    def __init__(self, data: str= 'diabetesbal.csv'):
        """From the infromation gotten from testing for the best algorithm and it's best parameters for this problem.
        I am creating a class for the diabetes prediction model

        Args:
            data (str, optional): Data to use for tarining and testing. Defaults to 'diabetesbal.csv'.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data = os.path.join(script_dir, "diabetes_model_files", data)
        self.helper = DiabetesHelper(self.data)
        self.model = LGBMClassifier(
            random_state=42, 
            verbose=-1,
            colsample_bytree=0.8,
            learning_rate = 0.01,
            max_depth = 10,
            n_estimators=200,
            num_leaves=21,
            subsample=0.8
            )
        
        # Variable to store the correct column order
        self.feature_columns = None
        
    def train_model(self):
        X_train, X_test, y_train, y_test = self.helper.get_processed_splits()
        
        # Save the column order from the training data
        self.feature_columns = X_train.columns.tolist()
        print(f"Model trained with columns: {self.feature_columns}") # Added for debugging
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.helper.metrics(y_test, y_pred)
        
    def predict(self, complete_data: dict, threshold: float = 0.5):
        """
            Predict whether a user has diabetes based on input features.

        Args:
            complete_data (dict): The input data.
            threshold (float, optional): The probability threshold to use for
                                         classifying "Diabetic". Defaults to 0.5.
        """
        data_copy = complete_data.copy()
        data_copy['gender_Male'] = 1 if data_copy.pop('gender') == 'Male' else 0  
        input_df = pd.DataFrame([data_copy])

        if self.feature_columns:
            X = input_df[self.feature_columns]
        else:
            raise ValueError("Model has not been trained yet.")
        probabilities = self.model.predict_proba(X)[0]
        prob_diabetic = probabilities[1]
        prediction = 1 if prob_diabetic >= threshold else 0
        
        return "Diabetic" if prediction == 1 else "Non-diabetic"