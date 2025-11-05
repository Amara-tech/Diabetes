import google.generativeai as genai
from textwrap import dedent
import json
import pandas as pd
import re


class PreprocessingGeneration:
    def __init__(self, model_name="gemini-2.5-pro"):
        self.model = genai.GenerativeModel(model_name)
        self.template = dedent("""
        You are a medical reasoning assistant helping prepare input data for a diabetes prediction model.

        ### Task:
        Your goal is to analyze the user's incomplete health data, use the retrieved context, and infer or suggest reasonable estimates for missing values.  
        Do not create random numbers; instead, use reasoning based on correlations, patterns, or standard health ranges described in the documents.

        ### Retrieved Context (Knowledge Base):
        {retrieved_docs}

        ### User-Provided Data:
        {user_data}

        ### Instructions:
        1. Identify which features are missing from the user data (e.g., glucose, BMI, blood pressure, insulin).
        2. For each missing feature, use the retrieved context to infer a realistic estimated range or value.  
           - Explain your reasoning briefly for each estimate.  
           - Reference insights from the retrieved documents if relevant.
        3. Fill the missing values with your best estimations.
        4. Return a structured JSON-like output that includes all fields (filled and original).

        ### Output Format Example:
        **Reasoning:**
        - Glucose is missing; based on user's BMI and age, similar cases in the documents show a likely fasting glucose range between 120-140 mg/dL. I'll estimate 130 mg/dL.
        - Insulin missing; average insulin for such glucose and BMI profile is around 80 µU/mL.

        **Completed Data:**
        {{
          "Age": 45,
          "Gender": "Female",
          "BMI": 31.2,
          "Blood Glucose": 130,
          "Heart disease": 1,
          "Hypertention": 0
        }}

        Now, output your reasoning and completed data in this format.
        """)

    def infer_missing_data(self, retrieved_docs, user_data):
        """Use Gemini to reason about and fill missing user health data."""
        retrieved_context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        prompt = self.template.format(
            retrieved_docs=retrieved_context,
            user_data=json.dumps(user_data, indent=2)
        )

        response = self.model.generate_content(prompt)
        text_output = response.text

        # Extract reasoning
        reasoning_match = re.search(r"\*\*Reasoning:\*\*(.*?)(?=\*\*Completed Data:|\Z)", text_output, re.S)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Reasoning not found."

        # --- Extract the JSON block under "Completed Data" ---
        json_match = re.search(r"\*\*Completed Data:\*\*([\s\S]*)", text_output)
        completed_json = json_match.group(1).strip() if json_match else "{}"

        # Clean and parse JSON safely
        try:
            completed_json = completed_json.strip("` \n")
            data_dict = json.loads(completed_json)
        except Exception:
            data_dict = {}

        # Convert to DataFrame
        df = pd.DataFrame([data_dict])

        return reasoning, df
    
    
    
    
class RecommendationGenerator:
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # Base prompt template
        self.template = dedent("""
        You are a health assistant specialized in diabetes prevention and management.

        Your task is to generate a clear, concise, and evidence-based health recommendation
        based on the user's information, the model's prediction, and retrieved medical documents.

        ### Context:
        The following context is retrieved from trusted medical sources:
        {retrieved_docs}

        ### User Information:
        {user_data}

        ### Prediction Result:
        The machine learning model classified this person as: **{prediction}**

        ### Your Task:
        1. Summarize what this prediction means for the user in simple, human-understandable language.
        2. Provide personalized lifestyle and dietary recommendations based on the context above.
        3. Highlight preventive or monitoring steps if the user is at risk.
        4. If the user is not at risk, include maintenance and wellness advice.
        5. Keep the tone informative, supportive, and medically responsible.

        ### Output Format:
        **Understanding Your Result:**  
        (Explain what their status means briefly)

        **Recommendations:**  
        - (List 3-5 actionable steps tailored to user data)

        **Next Steps:**  
        - (Include follow-up or monitoring advice)

        Now, write the full recommendation message.
        """)

    def generate(self, retrieved_docs, user_data, prediction):
        """Generate a health recommendation message."""
        context_text = "\n\n".join([doc["content"] for doc in retrieved_docs])

        prompt = self.template.format(
            retrieved_docs=context_text,
            user_data=user_data,
            prediction=prediction
        )

        response = self.model.generate_content(prompt)
        return response.text
    
    
    
    
