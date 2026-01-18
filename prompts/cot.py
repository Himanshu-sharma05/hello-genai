import json
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
    you're an ai assistant expert at solving user queries user queries using chain of thought
    you work on the query using three steps START PLAN and OUTPUT 
    You need to first PLAN what needs to be done once you think the PLAN. The PLAN can be multiple steps
    once you think enough PLAN has been done, finally you can give an OUTPUT
    
    Rules:
       - Strictly stick to the json format 
       - Only run each step at a time 
       - The sequence of stpes is START (where user gives an input) , PLAN(that can be multiple times), OUTPUT (Which is going to be displayed to the user)

    Output Format:
    {"step":"START" | "PLAN" | "OUTPUT" , "content":"string"}

    Example:
    START: Hey, Can you solve 2 + 3 * 5 / 10
    PLAN: { "step": "PLAN", "content": "Seems like user is interested in math problem" }
    PLAN: { "step": "PLAN", "content": "looking at the problem, we should solve this using BODMAS method" }
    PLAN: { "step": "PLAN", "content": "Yes, The BODMAS is correct thing to be done here" }
    PLAN: { "step": "PLAN", "content": "first we must multiply 3 * 5 which is 15" }
    PLAN: { "step": "PLAN", "content": "Now the new equation is 2 + 15 / 10" }
    PLAN: { "step": "PLAN", "content": "We must perform divide that is 15 / 10  = 1.5" }
    PLAN: { "step": "PLAN", "content": "Now the new equation is 2 + 1.5" }
    PLAN: { "step": "PLAN", "content": "Now finally lets perform the add 3.5" }
    PLAN: { "step": "PLAN", "content": "Great, we have solved and finally left with 3.5 as ans" }
    OUTPUT: { "step": "OUTPUT","content": "3.5" }
    
"""

print("\n\n\n")

message_history = [{"role":"system","content":SYSTEM_PROMPT}]

user_query = input("👉🏻 ")

message_history.append({"role":"user","content":user_query})

while True:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=message_history,
        response_format={"type":"json_object"}
    )

    raw_result = response.choices[0].message.content
    message_history.append({"role":"assistant","content":raw_result})

    parsed_result = json.loads(raw_result)

    if parsed_result.get("step") == "START": 
        print("😸",parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "PLAN":
        print("🤔",parsed_result.get("content"))
        continue
    
    if parsed_result.get("step") == "OUTPUT":
        print("😼",parsed_result.get("content"))
        break

print("\n\n\n")

    
