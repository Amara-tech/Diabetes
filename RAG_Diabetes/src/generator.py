import google.generativeai as genai
from textwrap import dedent
import json
import re


class PreprocessingGeneration:
    def __init__(self, model_name="gemini-2.5-pro"):
        self.model = genai.GenerativeModel(model_name)
        self.template = dedent("""
        You are a medical data inference assistant helping with a diabetes prediction system.

        Your task is to infer and complete **only the missing health inputs** required for predicting whether a user has diabetes.

        ### Purpose
        The goal of completing the data is to prepare full input values for a diabetes prediction model. Do not predict diabetes yourself — only fill in the missing fields logically based on the retrieved examples.

        ### Retrieved Patient Context
        {retrieved_docs}

        ### Current User Data (some fields may be missing)
        {user_data}
        
        ### User Query (contains extra unstructured details)
        {query}

        ### Instructions
        - Use patterns in the retrieved context AND the user query to infer realistic values...

        ### Instructions
        - Use patterns and correlations in the retrieved context to infer **realistic** values for any missing features.
        - If a value is already provided in the user data, keep it unchanged.
        - Only infer the following six features:
           - "age": <float>
           - "gender": "<string>"
           - "bmi": <float>
           - "hypertension": <0 or 1>
           - "heart_disease": <0 or 1>
           - "blood_glucose_level": <float>
        - Do not invent new fields.
        - Give concise, evidence-based reasoning for each inference.
        - Output must follow the exact format below.

        ### Response Format
            **Reasoning:**
               - Explain briefly how each field was inferred.

               **Completed Data:**
                   ```json
                 {{
                    "age": <float>,
                    "gender": "<string>",
                    "bmi": <float>,
                    "hypertension": <0 or 1>,
                    "heart_disease": <0 or 1>,
                    "blood_glucose_level": <float>
                 }}
        ---
            Make sure your final output includes BOTH sections exactly as shown:
            1. A "Reasoning" section.
            2. A "Completed Data" section containing a valid JSON object with only these six fields.
            Do not omit or summarize the JSON block.
 
        """)

    def infer_missing_data(self, retrieved_docs, user_data, query: str):
        """Use Gemini to reason about and fill missing user health data."""
        retrieved_context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        prompt = self.template.format(
            retrieved_docs=retrieved_context,
            user_data=json.dumps(user_data, indent=2),
            query=query 
        )

        # top-tier model for this kind of complex reasoning task.
        response = self.model.generate_content(prompt)
        text_output = response.text

        print("=== MODEL RAW OUTPUT ===")
        print(text_output)  
        
        #Get Reasoning
        reasoning_match = re.search(r"\*\*Reasoning:\*\*(.*?)(?=\*\*Completed Data:|\Z)", text_output, re.S)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Reasoning not found."

        # Geting  JSON
        # It looks for "Completed Data
        json_match = re.search(r"\*\*Completed Data:\*\*[\s\S]*?({[\s\S]*})", text_output)
        
        completed_json_str = None
        if json_match:
            completed_json_str = json_match.group(1).strip()
            # Clean up potential markdown backticks
            completed_json_str = completed_json_str.strip("` \n")
        
        # 3. Parse JSON
        data_dict = {}
        if completed_json_str:
            try:
                data_dict = json.loads(completed_json_str)
            except json.JSONDecodeError as e:
                print(f"Error parsing model's JSON output: {e}")
                print(f"Raw JSON string: {completed_json_str}")
                data_dict = {"error": "Failed to parse JSON from model"}
        else:
            print("Could not find 'Completed Data' JSON block in model output.")
            data_dict = {"error": "JSON block not found"}

        complete_data = {
            "reasoning": reasoning,
            "completed_data": data_dict
        }

        return complete_data


    
    
    
    
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
    
    
    
    
    
