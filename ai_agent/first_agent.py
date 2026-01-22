from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import requests
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_weather(city):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    if response:
        return f"The weather in {city} is {response.text}"
    return "something went wrong"


SYSTEM_PROMPT= '''
    You are an  ai agent who resolves user query using chain of thought.
    You Have access to tools from the tool list and can use them when needed
    You work on START PLAN AND OUTPUT steps.
    Once you think enough plan has been done you can give the output.
    For every tool call wait for the observe step which is output from the called tool

    Rules:
    - Strictly Follow the given JSON output format
    - Only run one step at a time.
    - The sequence of steps is START (where user gives an input), PLAN (That can be multiple times), OBSERVE (where you observe the output from the called tool) and finally OUTPUT (which is going to the displayed to the user).

    
    Output JSON Format:
    { "step": "START" | "PLAN" | "OUTPUT" | "TOOL", "content": "string", "tool": "string", "input": "string" 
    
    Available Tools:
    - get_weather(city) : it takes the city as the argument and returns the weather in that city

    Examples:
    Example 1:
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

    Example 2:
    START: What is the weather of Delhi?
    PLAN: { "step": "PLAN", "content": "Seems like user is interested in getting weather of Delhi in India" }
    PLAN: { "step": "PLAN", "content": "Lets see if we have any available tool from the list of available tools" }
    PLAN: { "step": "PLAN" ,"content": "Great, we have get_weather tool available for this query." }
    PLAN: { "step": "PLAN","content": "I need to call get_weather tool for delhi as input for city" }
    PLAN: { "step": "TOOL", "tool": "get_weather", "input": "delhi" }
    PLAN: { "step": "OBSERVE" ,"tool": "get_weather", "output": "The temp of delhi is cloudy with 20 C" }
    PLAN: { "step": "PLAN", "content": "Great, I got the weather info about delhi" }
    OUTPUT: { "step": "OUTPUT", "content": "The cuurent weather in delhi is 20 C with some cloudy sky." }

'''

user_query = input("🐰 :")
message_history = [{"role":"system","content":SYSTEM_PROMPT}]
message_history.append({"role":"user","content":user_query})

available_tools = {"get_weather":get_weather}
while True:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=message_history,
        response_format={"type":"json_object"}
    )

    raw_result = response.choices[0].message.content
    message_history.append({"role":"assistant","content":raw_result})
    parse_result = json.loads(raw_result)

    if parse_result.get("step") == 'START':
        print(f"🐻 : {parse_result.get("content")}")
        continue
    if parse_result.get("step") == 'PLAN':
        print(f"🤔 : {parse_result.get("content")}")
        continue
    if parse_result.get("step") == 'TOOL':
        tool_to_call = parse_result.get("tool")
        tool_input = parse_result.get("input")
        tool_response = available_tools[tool_to_call](tool_input)
        print(f"🛠️ Calling tool : {tool_to_call} with input :{tool_input}")
        message_history.append({"role":"developer","content":json.dumps({
            "step":"OBSERVE","tool":tool_to_call,"input":tool_input,"output":tool_response
        })})
        continue

    if parse_result.get("step") == 'OUTPUT':
        print(f"😼: {parse_result.get("content")}")
        break

print("\n\n\n")
