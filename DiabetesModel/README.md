Project Report: Diabetes Risk Prediction

This file summarizes the data processing and compares the performance of 10 machine learning models used to predict diabetes risk.

1. Project Goal

The primary goal was to build a risk assessment model. The model should predict a person's risk of having diabetes based on demographic/health factors and a single blood glucose reading (which can be obtained from an at-home meter).

The model avoids using the final diagnostic test (HbA1c_level) to remain a useful early-warning tool rather than a post-diagnosis tool.

2. Data Preprocessing

Several key steps were taken to prepare the data for modeling:

Undersampling: The highly imbalanced dataset was undersampled to create a balanced dataset.

Feature Selection (Dropping Columns):

smoking_history: Dropped after analysis showed it was a "noisy" and confounding variable, with its effects being driven by age.

HbA1c_level: This column was intentionally dropped. It is a long-term diagnostic test that confirms diabetes. Including it would be "cheating" as it is the final answer, not a risk factor.

blood_glucose_level: This column was intentionally kept. This was a key strategic decision. Unlike HbA1c_level, a single glucose reading is an indicator or symptom, not a final diagnosis. This feature provides significant predictive power without being the final diagnostic answer.

One-Hot Encoding:

The gender column was converted into numerical columns using pd.get_dummies(drop_first=True).

Feature Scaling:

A StandardScaler was used for algorithms sensitive to feature magnitude (e.g., KNN, Logistic Regression, SVM, and MLP).


3. Classification Model Comparison

All models were tuned using GridSearchCV to find their optimal hyperparameters.

1. Logistic Regression

Data Scaling Required? Yes.

Best Parameters:

--- GridSearch Results ---
Best parameters found: {'C': 0.01, 'penalty': 'l2', 'solver': 'saga'}
Best F1-score (on training data): 0.8348


Test Set Performance:

Accuracy: 79.40%
------------------------------
Confusion Matrix:
[[191  50]
 [ 53 206]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.79      0.79       241
           1       0.80      0.80      0.80       259

    accuracy                           0.79       500
   macro avg       0.79      0.79      0.79       500
weighted avg       0.79      0.79      0.79       500


2. K-Nearest Neighbors (KNN)

Data Scaling Required? Yes.

Best Parameters:

--- KNN GridSearch Results ---
Best parameters found: {'metric': 'minkowski', 'n_neighbors': 19, 'weights': 'uniform'}
Best F1-score (on training data): 0.8261


Test Set Performance:

Accuracy: 80.20%
------------------------------
Confusion Matrix:
[[182  59]
 [ 40 219]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.76      0.79       241
           1       0.79      0.85      0.82       259

    accuracy                           0.80       500
   macro avg       0.80      0.80      0.80       500
weighted avg       0.80      0.80      0.80       500


3. Decision Tree

Data Scaling Required? No.

Best Parameters:

--- Decision Tree DridSrach Results ---
Best parameters found are {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best F1-score (on training data): 0.8361248632329656


Test Set Performance:

Accuracy: 80.20%
------------------------------
Confusion Matrix:
[[157  84]
 [ 15 244]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.65      0.76       241
           1       0.74      0.94      0.83       259

    accuracy                           0.80       500
   macro avg       0.83      0.80      0.80       500
weighted avg       0.83      0.80      0.80       500


4. Random Forest

Data Scaling Required? No.

Best Parameters:

--- Random Forest GridSearch Results ---
Best parameters found: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
Best F1-score (on training data): 0.8508


Test Set Performance:

Accuracy: 81.40%
------------------------------
Confusion Matrix:
[[181  60]
 [ 33 226]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.75      0.80       241
           1       0.79      0.87      0.83       259

    accuracy                           0.81       500
   macro avg       0.82      0.81      0.81       500
weighted avg       0.82      0.81      0.81       500


5. Support Vector Machine (SVC)

Data Scaling Required? Yes.

Best Parameters:

--- SVM GridSearch Results ---
Best parameters found: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}
Best F1-score (on training data): 0.8370


Test Set Performance:

Accuracy: 80.20%
------------------------------
Confusion Matrix:
[[187  54]
 [ 45 214]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.81      0.78      0.79       241
           1       0.80      0.83      0.81       259

    accuracy                           0.80       500
   macro avg       0.80      0.80      0.80       500
weighted avg       0.80      0.80      0.80       500


6. XGBoost (eXtreme Gradient Boosting)

Data Scaling Required? No.

Best Parameters:

--- XGBoost GridSearch Results ---
Best parameters found: {'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 300, 'subsample': 1.0}
Best F1-score (on training data): 0.8532


Test Set Performance:

Accuracy: 82.20%
------------------------------
Confusion Matrix:
[[185  56]
 [ 33 226]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.77      0.81       241
           1       0.80      0.87      0.84       259

    accuracy                           0.82       500
   macro avg       0.83      0.82      0.82       500
weighted avg       0.82      0.82      0.82       500


7. LightGBM (Light Gradient Boosting Machine)

Data Scaling Required? No.

Best Parameters:

--- LightGBM GridSearch Results ---
Best parameters found: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 200, 'num_leaves': 21, 'subsample': 0.8}
Best F1-score (on training data): 0.8420


Test Set Performance:

Accuracy: 83.20%
------------------------------
Confusion Matrix:
[[191  50]
 [ 34 225]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.79      0.82       241
           1       0.82      0.87      0.84       259

    accuracy                           0.83       500
   macro avg       0.83      0.83      0.83       500
weighted avg       0.83      0.83      0.83       500


8. AdaBoost (Adaptive Boosting)

Data Scaling Required? No.

Best Parameters:

--- AdaBoost GridSearch Results ---
Best parameters found: {'learning_rate': 1.0, 'n_estimators': 300}
Best F1-score (on training data): 0.8530


Test Set Performance:

Accuracy: 81.60%
------------------------------
Confusion Matrix:
[[180  61]
 [ 31 228]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.75      0.80       241
           1       0.79      0.88      0.83       259

    accuracy                           0.82       500
   macro avg       0.82      0.81      0.81       500
weighted avg       0.82      0.82      0.81       500


9. Naive Bayes (GaussianNB)

Data Scaling Required? No.

Best Parameters:

-- Naive Bayes GridSearch Results ---
Best parameters found: {'var_smoothing': np.float64(4.641588833612782e-05)}
Best F1-score (on training data): 0.7902


Test Set Performance:

--- Final Naive Bayes Model Evaluation on Test Set ---
Accuracy: 79.40%
[[209  32]
 [ 71 188]]
              precision    recall  f1-score   support

           0       0.75      0.87      0.80       241
           1       0.85      0.73      0.78       259

    accuracy                           0.79       500
   macro avg       0.80      0.80      0.79       500
weighted avg       0.80      0.79      0.79       500


10. Multi-layer Perceptron (MLP)

Data Scaling Required? Yes.

Best Parameters:

--- MLP Neural Network GridSearch Results ---
Best parameters found: {'alpha': 0.001, 'hidden_layer_sizes': (50,), 'learning_rate_init': 0.001}
Best F1-score (on training data): 0.8409


Test Set Performance:

Accuracy: 82.00%
------------------------------
Confusion Matrix:
[[190  51]
 [ 39 220]]
------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.79      0.81       241
           1       0.81      0.85      0.83       259

    accuracy                           0.82       500
   macro avg       0.82      0.82      0.82       500
weighted avg       0.82      0.82      0.82       500


4. Performance Summary & Conclusion

Based on the test set results for all 10 tuned models, we can draw the following conclusions:

Key Metrics for Evaluation

For a medical risk model, we care most about two things:

Recall (Class 1): How many of the actual diabetes cases did we find? (Higher is better, we want to avoid false negatives).

Weighted F1-score: What is the best balance of precision and recall for both classes? (Higher is better).

Model Rankings

Best Overall Performer: LightGBM

This model achieved the highest Accuracy (83.20%) and the highest Weighted F1-score (0.83).

It also maintained a strong Class 1 Recall (0.87), meaning it correctly identified 87% of all diabetes cases. This combination of high accuracy and strong recall makes it the best all-around model for this problem.

Top Contenders (High Performance):

XGBoost: A very close second, with 82.20% Accuracy, 0.82 F1-score, and 0.87 Class 1 Recall.

MLP (Neural Network): Also a strong performer with 82.00% Accuracy, 0.82 F1-score, and a solid 0.85 Class 1 Recall.

Best Model for "Safety" (Highest Recall): Decision Tree

The simple Decision Tree achieved a Class 1 Recall of 0.94. This means it was the best model for not missing a diabetes case, finding 94% of all positive cases.

The Trade-off: This high recall came at the cost of low Class 1 Precision (0.74), meaning it produced more false positives (flagging 84 healthy people as at-risk). This is a classic "sensitivity vs. precision" trade-off.

Solid Mid-Pack Performers:

Random Forest, AdaBoost, KNN, and SVM all performed well, clustering around 80-82% Accuracy and 0.80-0.81 F1-scores. Their performance is strong and reliable.

Underperformers (Weakest Models):

Logistic Regression and Naive Bayes were the least effective models for this dataset, both with 79.4% accuracy.

Naive Bayes was the least suitable model for this task, as its Class 1 Recall was only 0.73. This means it missed 27% of all diabetes cases, which is unacceptably high for a medical risk model.

Final Conclusion

The advanced ensemble methods (LightGBM, XGBoost) and the MLP neural network provided the best balance of accuracy and recall. LightGBM stands out as the top-performing model.

If the project's main goal is to never miss a potential case (and you are willing to accept more false positives), the Decision Tree model would be the best choice due to its outstanding 94% recall.