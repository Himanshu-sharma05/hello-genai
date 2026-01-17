import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
# in zero shot prompting the model is given direct question or task without prior example
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
SYSTEM_PROMPT="your name is alex and you can only answer coding related questions or messages answer sorry for any other kind of question"
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"Hey there what's your name and what's variables"}]
)

print(response.choices[0].message.content)