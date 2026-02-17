import streamlit as st
import pandas as pd
import markdown
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import io


def generate_pdf(completed_data, prediction, advice):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Diabetes Assessment Report</b>", styles['Title']))
    story.append(Spacer(1, 12))

    # User Data Table
    data_items = [["Field", "Value"]]
    for key, value in completed_data.items():
        data_items.append([str(key), str(value)])

    table = Table(data_items)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT')
    ]))
    story.append(table)
    story.append(Spacer(1, 18))

    # Prediction
    story.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles['Heading2']))
    story.append(Spacer(1, 12))

    # Advice
    story.append(Paragraph("<b>Recommendation:</b>", styles['Heading2']))
    html_text = markdown.markdown(advice)
    story.append(Paragraph(html_text, styles['BodyText']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer



# IMPORT HIGH-LEVEL MODULES
from prepos import Preprocessing
from recommed import Recommender
from diabetes_model_files.custom_model import CustomModel

# CONFIGURATION & SETUP
st.set_page_config(page_title="Diabetes Assistant", layout="centered")

# RESET HANDLER - MUST RUN BEFORE FORM IS RENDERED
if st.session_state.get("reset_triggered", False):
    keys_to_clear = [
        'gender_input', 'age_input', 'bmi_input', 'bg_input', 
        'ht_input', 'hd_input', 'extra_input',
        'completed_data', 'prediction'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state['reset_triggered'] = False
    st.rerun()

# Initialize session state variables
st.session_state.setdefault('completed_data', None)
st.session_state.setdefault('prediction', None)


# LOAD ALL MODELS (cached)
@st.cache_resource
def load_all_models():
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

ml_model, prepro_pipeline, reco_pipeline = load_all_models()

# APP INTERFACE
st.title("Intelligent Diabetes Prediction Assistant")
st.write("This system uses AI to infer missing data, predict diabetes risk, and offer guidance.")

st.header("Enter Your Health Information")
st.write("Please fill in what you know. Our AI will help infer any missing values.")

# MAIN FORM

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
        blood_glucose_level = st.number_input(
            "Blood Glucose (Optional)", value=None, placeholder="100", key="bg_input"
        )

    st.write("Do you have a known diagnosis for the following?")
    col5, col6 = st.columns(2)
    with col5:
        hypertension_str = st.radio("Hypertension", ["No", "Yes"], index=None, horizontal=True, key="ht_input")
    with col6:
        heart_disease_str = st.radio("Heart Disease", ["No", "Yes"], index=None, horizontal=True, key="hd_input")

    extra_info_query = st.text_area(
        "Add relevant information (Optional)",
        placeholder="I am 5'4\" tall, I exercise twice a week...",
        key="extra_input"
    )

    submit_col, reset_col = st.columns([3, 1])
    with submit_col:
        submitted = st.form_submit_button("Analyze My Risk", use_container_width=True, type="primary")
    with reset_col:
        reset_pressed = st.form_submit_button("Reset Form", use_container_width=True)

# RESET LOGIC
if reset_pressed:
    st.session_state['reset_triggered'] = True
    st.rerun()


# FORM PROCESSING
if submitted:
    if not gender or not age:
        st.error("Please fill in the required fields (Gender and Age).")
    else:
        user_data_dict = {
            "gender": gender,
            "age": float(age),
            "bmi": bmi,
            "hypertension": 1 if hypertension_str == "Yes" else 0,
            "heart_disease": 1 if heart_disease_str == "Yes" else 0,
            "blood_glucose_level": blood_glucose_level
        }

        try:
            # Use RAG inference if text info provided
            if extra_info_query:
                with st.spinner("Analyzing extra info and inferring missing data..."):
                    inferred_result = prepro_pipeline.recommend(
                        query=extra_info_query,
                        user_data_dict=user_data_dict
                    )
                completed_data = inferred_result["completed_data"]

                st.subheader("Inferred Health Profile")
                st.dataframe(pd.DataFrame([completed_data]), hide_index=True)
            else:
                completed_data = user_data_dict

            # Prediction
            with st.spinner("Running ML risk assessment..."):
                prediction = ml_model.predict(completed_data, threshold=0.35)

            st.session_state['completed_data'] = completed_data
            st.session_state['prediction'] = prediction

            # Display Prediction
            st.subheader("Your Diabetes Risk Profile")
            if prediction == "Non-diabetic":
                st.success(f"Prediction: **{prediction}**")
            else:
                st.error(f"Prediction: **{prediction}**")

            # Advice
            with st.spinner("Generating advice..."):
                initial_advice = reco_pipeline.advice(
                    user_data=completed_data,
                    prediction=prediction
                )

            st.subheader("Recommendation")
            st.markdown(initial_advice)
            pdf_buffer = generate_pdf(
                            completed_data=st.session_state['completed_data'],
                            prediction=st.session_state['prediction'],
                            advice=initial_advice
)

            st.download_button(
                label="Download Results as PDF",
                data=pdf_buffer,
                file_name="diabetes_report.pdf",
                mime="application/pdf"
)

        except Exception as e:
            st.error("You did not provide complete data. Fill in the extra information box if unsure.")


# RECOMMENDATION FLOW

if st.session_state['prediction'] is not None:
    st.divider()
    st.header("Deep Dive")
    st.write("Ask follow-up questions based on your results.")

    with st.container():
        col_q, col_btn = st.columns([4, 1])
        with col_q:
            specific_question = st.text_input(
                "Ask a specific question",
                placeholder="E.g., What should I eat for breakfast?",
                label_visibility="collapsed"
            )
        with col_btn:
            ask_button = st.button("Ask AI", use_container_width=True, type="primary")

        if ask_button:
            if specific_question:
                with st.spinner("Thinking..."):
                    recommendation = reco_pipeline.recommend(
                        query=specific_question,
                        user_data=st.session_state['completed_data'],
                        prediction=st.session_state['prediction']
                    )
                st.markdown("AI Response")
                st.markdown(recommendation)
            else:
                st.warning("Please type a question first.")

st.divider()
st.caption("This project is dedicated to Our Lady of Perpetual Help.")
