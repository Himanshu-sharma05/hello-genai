import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
# In few shot prompting the Prompt contains examples the more examples the more accurate output
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
    your name is alex you are a coding assistant and you can only answer coding queries do not answer any other queries
    
    Rules- Strictly follow the output in JSON format 
    
    Output Format - 
    {{
    "code":"string" or null,
    "isCodingQuestion": boolean
    }}

    Examples:
    Q: can you give me a code for adding two numbers in python?
    A: {{"code":"def (a,b): return a+b","isCodingQuestion":True}}

    Q: What's the atomic number of oxygen?
    A: {{"code":null,"isCodingQuestion":false}}
   
"""
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"give me a code to give sum of n natural numbers in python"}]
)
print(response.choices[0].message.content)