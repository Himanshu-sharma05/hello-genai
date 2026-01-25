from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role":"user","content":[
        {"type":"text","text":"Generate a caption for the provided image in 50 words"},
        {"type":"image_url","image_url":{"url":"https://images.pexels.com/photos/2098406/pexels-photo-2098406.jpeg"}}]}]
)

print(response.choices[0].message.content)