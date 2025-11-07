from .helpers import DiabetesHelper
from lightgbm import LGBMClassifier
import pandas as pd

class CustomModel:
    def __init__(self, data: str= 'diabetesbal.csv'):
        """From the infromation gotten from testing for the best algorithm and it's best parameters for this problem.
        I am creating a class for the diabetes prediction model

        Args:
            data (str, optional): Data to use for tarining and testing. Defaults to 'diabetesbal.csv'.
        """
        self.data = data
        self.helper = DiabetesHelper(self.data)
        self.model = LGBMClassifier(
            random_state=42, # Parameters were chosen based on the results gotten from gridsearch best parameters
            # Best parameters found: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 200, 'num_leaves': 21, 'subsample': 0.8}
            verbose=-1,
            colsample_bytree=0.8,
            learning_rate = 0.01,
            max_depth = 10,
            n_estimators=200,
            num_leaves=21,
            subsample=0.8
            )
        
    def train_model(self):
        X_train, X_test, y_train, y_test = self.helper.get_processed_splits()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.helper.metrics(y_test, y_pred)
        
    def predict(self, complete_data: dict):
        """
            Predict whether a user has diabetes based on input features.

        Args:
        complete_data (dict): Example:
            {
                'age': 45,
                'gender': 'Male',
                'bmi': 32.1,
                'hypertension': 1,
                'heart_disease': 0,
                'blood_glucose_level': 155
            }

        Returns:
           str: 'Diabetic' or 'Non-diabetic'
        """
        complete_data['gender_Male'] = 1 if complete_data.pop('gender') == 'Male' else 0  
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([complete_data])

        # Apply the same preprocessing as the training data
        X = input_df

        # Get prediction (0 or 1)
        prediction = self.model.predict(X)[0]

        # Convert numeric result to human-readable label
        return "Diabetic" if prediction == 1 else "Non-diabetic"
