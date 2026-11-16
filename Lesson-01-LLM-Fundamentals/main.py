
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Send a question to the LLM
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="How many leave days does ABC Company provide?"
)

# Display the response
print(response.text)
