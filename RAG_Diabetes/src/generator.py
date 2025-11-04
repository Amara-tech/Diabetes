import google.generativeai as genai
from textwrap import dedent

class RecommendationGenerator:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # Base prompt template
        self.template = dedent("""
        You are a health assistant specialized in diabetes prevention and management.

        Your task is to generate a clear, concise, and evidence-based health recommendation
        based on the user’s information, the model’s prediction, and retrieved medical documents.

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
        - (List 3–5 actionable steps tailored to user data)

        **Next Steps:**  
        - (Include follow-up or monitoring advice)

        Now, write the full recommendation message.
        """)

    def generate(self, retrieved_docs, user_data, prediction):
        """Generate a health recommendation message."""
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = self.template.format(
            retrieved_docs=context_text,
            user_data=user_data,
            prediction=prediction
        )

        response = self.model.generate_content(prompt)
        return response.text
