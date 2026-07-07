from google import genai
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
def generate_completion(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text