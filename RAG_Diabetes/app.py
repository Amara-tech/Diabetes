import streamlit as st
import pandas as pd

# IMPORT  HIGH-LEVEL MODULES
from prepos import Preprocessing
from recommed import Recommender
from diabetes_model_files.custom_model import CustomModel

# CONFIGURATION & SETUP 
st.set_page_config(page_title="Diabetes Assistant", layout="centered")

# Initialize session state variables if they don't exist
if 'completed_data' not in st.session_state:
    st.session_state['completed_data'] = None
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

# LOAD YOUR THREE MAIN MODELS
@st.cache_resource
def load_all_models():
    """
    Initializes the ML model and the two RAG pipelines.
    This runs once and is cached by Streamlit.
    """
    # Use st.spinner during initial load so the user knows something is happening
    with st.spinner("Loading AI Models... please wait."):
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

#  MAIN FORM
st.header("Enter Your Health Information")
st.write("Please fill in what you know. Our AI will help infer any missing values.")

with st.form("data_form"):
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True, index=None, key="gender_input")
    with col2:
        age = st.number_input("Age", min_value=5, max_value=100, value=None, placeholder="Your age", key="age_input")

    col3, col4 = st.columns(2)
    with col3:
        bmi = st.number_input("BMI (Optional)", value=None, placeholder="20.0", key="bmi_input")
    with col4:
        blood_glucose_level = st.number_input("Blood Glucose (Optional)", value=None, placeholder="100", key="bg_input")

    st.write("Do you have a known diagnosis for the following?")
    col5, col6 = st.columns(2)
    with col5:
        hypertension_str = st.radio("Hypertension", ["No", "Yes"], index=None, horizontal=True, key="ht_input")
    with col6:
        heart_disease_str = st.radio("Heart Disease", ["No", "Yes"], index=None, horizontal=True, key="hd_input")

    extra_info_query = st.text_area(
        "Add relevant information (Optional)",
        placeholder="I am 5'4\" tall, I exercise twice a week, I feel tired often...",
        key="extra_input"
    )

    submit_col, reset_col = st.columns([3, 1])
    with submit_col:
        submitted = st.form_submit_button("Analyze My Risk", use_container_width=True, type="primary")
    with reset_col:
        reset_pressed = st.form_submit_button("Reset Form", use_container_width=True)

#  FORM PROCESSING 
if reset_pressed:
    # Clear specific session state keys and rerun immediately
    keys_to_clear = [
        'gender_input', 'age_input', 'bmi_input', 'bg_input', 
        'ht_input', 'hd_input', 'extra_input',
        'completed_data', 'prediction'
    ]
    # Delete them from session state if they exist
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Rerun the app to reflect the cleared state
    st.rerun()

if submitted:
    # Validate required inputs
    if not gender or not age:
        st.error("Please fill in the required fields (Gender and Age).")
    else:
        # Build the initial data dict
        user_data_dict = {
            "gender": gender,
            "age": float(age),
            "bmi": bmi,
            "hypertension": 1 if hypertension_str == "Yes" else 0,
            "heart_disease": 1 if heart_disease_str == "Yes" else 0,
            "blood_glucose_level": blood_glucose_level
        }
        
        try:
            # PATH 1: User provided extra textual info -> Use RAG for inference
            if extra_info_query:
                with st.spinner("Analyzing extra info and inferring missing data..."):
                    inferred_result = prepro_pipeline.recommend(
                        query=extra_info_query, 
                        user_data_dict=user_data_dict
                    )
                    completed_data = inferred_result["completed_data"]

                st.subheader("Inferred Health Profile")
                st.info("We filled in missing values based on the details you provided.")
                st.dataframe(pd.DataFrame([completed_data]), hide_index=True)

            else:
                completed_data = user_data_dict

            #  ML PREDICTION
            with st.spinner("Running ML risk assessment..."):
                prediction = ml_model.predict(completed_data, threshold=0.35)
            
            # SAVE TO SESSION STATE
            st.session_state['completed_data'] = completed_data
            st.session_state['prediction'] = prediction

            # DISPLAY RESULTS
            st.subheader("Your Diabetes Risk Profile")
            if prediction == "Non-diabetic":
                st.success(f"Prediction is: **{prediction}**")
            else:
                st.error(f"Prediction is: **{prediction}**")
            
            # Initial Advice
            with st.spinner("Generating initial advice..."):
                initial_advice = reco_pipeline.advice(
                    user_data=completed_data,
                    prediction=prediction
                )
                st.subheader("Summary Recommendation")
                st.markdown(initial_advice)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            # Optional: print full traceback to terminal for debugging
            
            
# FOLLOW-UP RECOMMENDATION FLOW
# This section runs if a prediction exists in session state, 
# regardless of whether the form was just submitted or not.
if st.session_state['prediction'] is not None:
    st.divider()
    st.header("Deep Dive")
    st.write("Ask follow-up questions based on your results.")

    # Using a container or expander can help organize this section
    with st.container():
        col_q, col_btn = st.columns([4, 1])
        with col_q:
            specific_question = st.text_input(
                "Ask a specific question",
                placeholder="E.g., What specifically should I eat for breakfast?",
                label_visibility="collapsed"
            )
        with col_btn:
            ask_button = st.button("Ask AI", use_container_width=True, type="primary")

        if ask_button and specific_question:
            with st.spinner("Thinking..."):
                recommendation = reco_pipeline.recommend(
                    query=specific_question,
                    user_data=st.session_state['completed_data'],
                    prediction=st.session_state['prediction']
                )
                st.markdown("AI Response")
                st.markdown(recommendation)
        elif ask_button and not specific_question:
            st.warning("Please type a question first.")

st.divider()
st.caption("This project is dedicated to Our Lady of Perpetual Help.")