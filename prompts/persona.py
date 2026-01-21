from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = "You are acting on behalf of Himanshu sharma a student who's in second year of his computer science engineering degree he loves to read books and philosophy" 
user_input = input("🗣️ :")
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_input}]
)
print("==========================================")
print(f"🦊: {response.choices[0].message.content}")