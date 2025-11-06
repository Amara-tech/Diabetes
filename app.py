# app.py
import streamlit as st
import pandas as pd
from RAG_Diabetes.prepos import Preprocessing
from DiabetesModel.light_gbm import DiabetesPredictor
from src.recommed import RAGRecommender

# Streamlit Page Config
st.set_page_config(
    page_title="Diabetes RAG Assistant",
    page_icon="💊",
    layout="wide"
)

# --- Sidebar: Dedication ---
with st.sidebar:
    st.title("🕊️ Dedication")
    st.markdown("""
    **Dedicated to Our Lady of Perpetual Help**,  
    whose intercession guides this work of compassion,  
    healing, and scientific discovery.

    ---
    **Developed by:** Helen Oba  
    **Year:** 2025
    """)

# --- App Title ---
st.title("🩺 AI Diabetes Risk & Recommendation Assistant")
st.markdown("""
Welcome!  
This intelligent assistant:
1. **Fills in missing medical data**,  
2. **Predicts diabetes likelihood**, and  
3. **Provides personalized recommendations** based on your results.

---
""")

# --- Input Form ---
with st.form("user_input_form"):
    st.subheader("Enter Patient Information")
    st.info("Leave a field empty if unknown — the AI Preprocessing model will infer it.")

    age = st.text_input("Age (years)", "")
    gender = st.selectbox("Gender", ["", "Male", "Female"])
    bmi = st.text_input("BMI", "")
    hypertension = st.selectbox("Hypertension", ["", "Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", ["", "Yes", "No"])

    submitted = st.form_submit_button("Run Analysis")

if submitted:
    st.markdown("## 🧩 Step 1️⃣: Data Preprocessing")
    with st.spinner("Using LLM-powered Preprocessing RAG to complete missing values..."):
        user_data = {
            "Age": age or None,
            "Gender": gender or None,
            "BMI": bmi or None,
            "Hypertension": hypertension or None,
            "HeartDisease": heart_disease or None,
        }

        preprocessor = Preprocessing(docs_path="RAG_DOCs/Prepros")
        completed_df = preprocessor.process(user_data)  # Returns a pandas DataFrame
        st.success("Data completed successfully!")

        st.markdown("### 🧾 Completed Data:")
        st.dataframe(completed_df)

    # Step 2️⃣: Prediction
    st.markdown("## ⚙️ Step 2️⃣: Predicting Diabetes Risk")
    with st.spinner("Predicting using your trained Diabetes model..."):
        predictor = DiabetesPredictor()
        prediction_value = predictor.predict(completed_df)  # returns 0 or 1

        # Interpret result
        prediction_label = "Yes" if prediction_value == 1 else "No"
        st.success("Prediction complete!")
        st.markdown(f"**Predicted Diabetes:** `{prediction_label}`")

    # Step 3️⃣: Recommendation RAG
    st.markdown("## 💡 Step 3️⃣: Personalized Recommendations")
    with st.spinner("Retrieving personalized recommendations using RAG..."):
        rag = RAGRecommender()
        query = "Provide recommendations for managing diabetes risk based on this user's data."
        response = rag.recommend(query, completed_df.to_dict(orient='records')[0], prediction_label)
        st.success("Recommendations ready!")

        st.markdown("### Recommendations:")
        st.write(response)

# --- Footer ---
st.markdown("""
---
🙏 *Dedicated to Our Lady of Perpetual Help —  
for her guidance and compassion that inspire this work.*  
""")
