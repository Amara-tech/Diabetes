---
title: Diabetes RAG Recommender
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: RAG_Diabetes/app.py
pinned: false
---

# Diabetes Risk Prediction with RAG-Powered Recommendation System

This project predicts a person’s risk of having diabetes using demographic and health data, and enhances the user experience with a **Retrieval-Augmented Generation (RAG)** model for intelligent preprocessing and personalized recommendations.

---

## Project Overview

The goal of this project is to build a **diabetes risk assessment system** that can:
1. Predict diabetes risk based on user health inputs.
2. Handle incomplete data by using a RAG model to reason and estimate missing values (e.g., blood glucose).
3. Provide personalized lifestyle or prevention recommendations for both “At-Risk” and “Not At-Risk” individuals.

The system combines **machine learning** for risk prediction and **RAG (Retrieval-Augmented Generation)** for context-aware reasoning and advice.

---

## Core Components

### 1. **Base Machine Learning Model**
- Built to predict diabetes risk using structured features such as:
  - Age, BMI, Gender, Blood Glucose Level, etc.
- Multiple algorithms were tested and compared, including:
  - Logistic Regression, KNN, Decision Tree, Random Forest, SVM, XGBoost, LightGBM, AdaBoost, Naive Bayes, and MLP.
- The **LightGBM** model achieved the best overall performance:
  - **Accuracy:** 83.2%
  - **Weighted F1-Score:** 0.83
  - **Recall (Positive class):** 0.87

---

### 2. **RAG Module**
The RAG model operates in two key phases:
1. **Preprocessing (Input Reasoning):**
   - Uses medical and diagnostic documents to infer missing data logically (e.g., estimating blood glucose patterns).
   - Documents include:
     - WHO Diagnostic Guidelines  
     - Pathophysiology of Diabetes
2. **Recommendation (Post-Prediction Guidance):**
   - Provides tailored advice and explanations based on prediction outcomes.
   - Documents include:
     - *Health Box Diabetes* PDF  
     - Lifestyle and prevention guides

RAG Framework: **LangChain**

---

### 3. **User Interface (Streamlit)**
A simple, interactive **Streamlit app** allows users to:
- Input their health information (age, gender, BMI, etc.)
- Automatically preprocess incomplete inputs using the RAG layer
- View the model’s prediction
- Receive personalized advice and explanations

---

## Technologies Used

| Component | Technology |
|------------|-------------|
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| RAG Framework | LangChain |
| Vector Store | ChromaDB (planned) |
| User Interface | Streamlit |
| Data Handling | pandas, numpy |
| Environment | Python 3.10+ |

---

## How to Run

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/Amara-tech/Diabetes.git](https://github.com/Amara-tech/Diabetes.git)
   cd Diabetes/RAG_Diabetes