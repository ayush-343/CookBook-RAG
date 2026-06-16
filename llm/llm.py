import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", "http://127.0.0.1:1234/v1/"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio")
)
def ask_llm(context, question):
    prompt = f"""
            You are a helpful recipe assistant.

            Instructions:

            - Use ONLY the provided context.
            - If multiple recipes match the request, choose the most complete one.
            - If the answer is not present in the context, say:
            "I don't know based on the document."
            - Do not invent ingredients.
            - Summarize recipes clearly.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL", "nvidia/nemotron-3-nano-4b"),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content