import streamlit as st
import pandas as pd
import json

# IMPORT YOUR HIGH-LEVEL MODULES
from prepos import Preprocessing
from recommed import Recommender
from diabetes_model_files.custom_model import CustomModel

# LOAD YOUR THREE MAIN MODELS
@st.cache_resource
def load_all_models():
    """
    Initializes the ML model and the two RAG pipelines.
    This runs once and is cached by Streamlit.
    """
    print("Loading ML model...")
    ml_model = CustomModel()
    ml_model.train_model()
    
    print("Loading Preprocessing RAG Pipeline...")
    prepro_pipeline = Preprocessing() 
    
    print("Loading Recommendation RAG Pipeline...")
    reco_pipeline = Recommender()

    print("All models loaded successfully.")
    
    return ml_model, prepro_pipeline, reco_pipeline

# Load all models into memory
ml_model, prepro_pipeline, reco_pipeline = load_all_models()


# APP INTERFACE
st.title("Intelligent Diabetes Prediction Assistant")
st.write("This system uses AI to infer missing data, predict diabetes risk, and offer guidance.")
st.header("Enter Your Health Information")
st.write("Please fill in what you know. Our AI will help infer any missing values.")

with st.form("data_form"):
    
    col1, col2 = st.columns(2)
    with col1:
        # ADD key="gender"
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True, index=None, key="gender")
    with col2:
        # ADD key="age"
        age = st.number_input("Age", min_value=5, max_value=100, value=None, placeholder="Your age", key="age")

    col3, col4 = st.columns(2)
    with col3:
        # ADD key="bmi"
        bmi = st.number_input("BMI (Optional)", value=None, placeholder="20.0", key="bmi")
    with col4:
        # ADD key="blood_glucose_level"
        blood_glucose_level = st.number_input("Blood Glucose (Optional)", value=None, placeholder="100", key="blood_glucose_level")

    st.write("Do you have a known diagnosis for the following?")
    col5, col6 = st.columns(2)
    with col5:
        # ADD key="hypertension_str"
        hypertension_str = st.radio("Hypertension", ["No", "Yes"], index=None, horizontal=True, key="hypertension_str")
    with col6:
        # ADD key="heart_disease_str"
        heart_disease_str = st.radio("Heart Disease", ["No", "Yes"], index=None, horizontal=True, key="heart_disease_str")

    # ADD key="extra_info_query"
    extra_info_query = st.text_area(
        "Add relevant information (Optional)",
        placeholder="I am 5'4\" tall, I exercise twice a week, I feel tired often...",
        key="extra_info_query"
    )

    submit_col, reset_col = st.columns([3, 1])

    with submit_col:
        submitted = st.form_submit_button("Analyze My Risk", use_container_width=True)
    
    with reset_col:
        # This button is now also a submit button, but we'll check for it
        reset_pressed = st.form_submit_button("Reset", use_container_width=True)

#PROGRAM FLOW 
if submitted:
    
    # READ from st.session_state
    if not st.session_state.gender or not st.session_state.age:
        st.error("Please fill in the required fields (Gender and Age).")
    else:
        # Build the input dict from st.session_state
        user_data_dict = {
            "gender": st.session_state.gender,
            "age": float(st.session_state.age),
            "bmi": st.session_state.bmi,
            "hypertension": 1 if st.session_state.hypertension_str == "Yes" else 0,
            "heart_disease": 1 if st.session_state.heart_disease_str == "Yes" else 0,
            "blood_glucose_level": st.session_state.blood_glucose_level
        }
        
        #  Preprocessing RAG
        with st.spinner("Analyzing and inferring missing data..."):
            if st.session_state.extra_info_query:
                try:
                    inferred_result = prepro_pipeline.recommend(
                    query=st.session_state.extra_info_query, 
                    user_data_dict=user_data_dict
                )
                    completed_data = inferred_result["completed_data"]

                    st.subheader("Inferred Health Profile")
                    st.write("Filled missing Values based on reasoning on some consistent data")
                    st.dataframe(pd.DataFrame([completed_data]))

                    # ML Model Prediction
                    with st.spinner("Running ML prediction..."):
                        prediction = ml_model.predict(completed_data, threshold=0.35)
                
                    st.subheader("Your Diabetes Risk Profile")
                    if prediction == "Non-diabetic":
                        st.success(f"Prediction is: **{prediction}**")
                    else:
                        st.error(f"Prediction is: **{prediction}**")
                    # Store results for the next step
                    st.session_state['completed_data'] = completed_data
                    st.session_state['prediction'] = prediction
                    reccommed = reco_pipeline.advice(
                        user_data=user_data_dict,
                        prediction=prediction
                )
                    st.subheader("Recommendation")
                    st.markdown(reccommed)
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")    
            else:
                 with st.spinner("Running ML prediction..."):
                        prediction = ml_model.predict(user_data_dict, threshold=0.35)
                        st.subheader("Your Diabetes Risk Profile")
                        if prediction == "Non-diabetic":
                            st.success(f"Prediction is: **{prediction}**")  
                        else:
                            st.error(f"Prediction is: **{prediction}**")    
                        reccommed = reco_pipeline.advice(
                        user_data=user_data_dict,
                        prediction=prediction
                        )         
if reset_pressed:
    # If the reset button was clicked, clear session state
    if 'completed_data' in st.session_state:
        del st.session_state['completed_data']
    if 'prediction' in st.session_state:
        del st.session_state['prediction']
    st.rerun()
    
# RECOMMENDATION FLOW
if 'prediction' in st.session_state:
    st.header("Get Personalized Recommendations")
    
    specific_question = st.text_input(
        "Ask a specific question about your results",
        placeholder="What kind of foods should I eat? What exercise is best?"
    )
    
    if st.button("Get Advice"):
        if not specific_question:
            st.error("Please ask a question.")
        else:
            with st.spinner("Generating your recommendation..."):
                
                #Recommendation RAG
                recommendation = reco_pipeline.recommend(
                    query=specific_question,
                    user_data=st.session_state['completed_data'],
                    prediction=st.session_state['prediction']
                )
                
                st.markdown(recommendation)
st.divider()
st.caption("This project is dedicated to Our Lady of Perpetual Help.")                
                           
                