import pandas as pd 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DiabetesHelper:
    
    def __init__(self, file_path):
        """
        Loads the raw data ONCE and stores it.
        """
        self.raw_df = pd.read_csv(file_path)
        print("Helper initialized and data loaded.")
        
    def get_processed_splits(self):
        """
        This function creates fresh X, y, and splits every time
        by operating on a COPY of the raw data.
        """
        
        # Start with a fresh copy of the raw data
        df = self.raw_df.copy() 
        
        # Define columns to drop
        cols_to_drop = ['smoking_history']
        
        # Check if 'Unnamed: 0' exists before trying to drop it
        if 'Unnamed: 0' in df.columns:
            cols_to_drop.append('Unnamed: 0')
            
        df = df.drop(columns=cols_to_drop) # No inplace=True
        
        # One-hot encode gender
        df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype=int)
        
        # Define X and y
        X = df.drop(columns=['diabetes', 'HbA1c_level'])
        y = df['diabetes']
        
        # Return the splits
        return train_test_split(X, y, test_size=0.25, random_state=42)
    
    @staticmethod
    def scaling(X_train, X_test):
        """
        Scales the data. This is a "static" method because
        it doesn't need to access 'self'.
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    @staticmethod
    def metrics(y_test, y_pred):
        """
        Prints all relevant metrics. Also a static method.
        """
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("-" * 30)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 30)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))