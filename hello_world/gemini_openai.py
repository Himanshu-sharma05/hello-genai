import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role":"system","content":"You are a maths expert and can only answer maths related query answer sorry for any other kind of query"},{"role":"user",
        "content":"what's a proxy battle"}]
)
print(response.choices[0].message.content)