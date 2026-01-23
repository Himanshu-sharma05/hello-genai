from dotenv import load_dotenv
from openai import OpenAI
import os 
from pydantic import BaseModel,Field
from typing import Optional
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def run_command(cmd):
    result = os.system(cmd)
    return result


SYSTEM_PROMPT = """
    You are an ai agent who is an expert in solving user query using chain of thought.
    You work on four steps START , PLAN , OBSERVE ,TOOL, OUTPUT
    In PLAN step you first plan what to do and think (plan can be multiple) then you call the tool if needs and OBSERVE the output of the tool 
    In TOOL step you think which tool to use by checking available tools and what input to give to the tool 
    Once you think all the neccessary plan and observe steps are done you can give OUTPUT

    RULES:
        - Strictly return output in JSON format
        - execute one step at a time
        - The sequence of steps is START(taking the user input) , PLAN(can be multiple) , OBSERVE (if needed) , OUTPUT 

    OUTPUT FORMAT:
        - {"step":"START" | "PLAN"| "OBSERVE" | "OUTPUT" | "TOOL" , "content" : "string" , "tool": "string", "input":"string"}
AVAILABLE TOOLS:
        - run_command(cmd:string) : Takes a linux command as input and run it on the system then returns the output of the command

    EXAMPLES:
        Example 1:
    START: Hey, Can you solve 2 + 3 * 5 / 10
    PLAN: { "step": "PLAN": "content": "Seems like user is interested in math problem" }
    PLAN: { "step": "PLAN": "content": "looking at the problem, we should solve this using BODMAS method" }
    PLAN: { "step": "PLAN": "content": "Yes, The BODMAS is correct thing to be done here" }
    PLAN: { "step": "PLAN": "content": "first we must multiply 3 * 5 which is 15" }
    PLAN: { "step": "PLAN": "content": "Now the new equation is 2 + 15 / 10" }
    PLAN: { "step": "PLAN": "content": "We must perform divide that is 15 / 10  = 1.5" }
    PLAN: { "step": "PLAN": "content": "Now the new equation is 2 + 1.5" }
    PLAN: { "step": "PLAN": "content": "Now finally lets perform the add 3.5" }
    PLAN: { "step": "PLAN": "content": "Great, we have solved and finally left with 3.5 as ans" }
    OUTPUT: { "step": "OUTPUT": "content": "3.5" }

    Example 2:
    START: What is the weather of Delhi?
    PLAN: { "step": "PLAN": "content": "Seems like user is interested in getting weather of Delhi in India" }
    PLAN: { "step": "PLAN": "content": "Lets see if we have any available tool from the list of available tools" }
    PLAN: { "step": "PLAN": "content": "Great, we have get_weather tool available for this query." }
    PLAN: { "step": "PLAN": "content": "I need to call get_weather tool for delhi as input for city" }
    PLAN: { "step": "TOOL": "tool": "get_weather", "input": "delhi" }
    PLAN: { "step": "OBSERVE": "tool": "get_weather", "output": "The temp of delhi is cloudy with 20 C" }
    PLAN: { "step": "PLAN": "content": "Great, I got the weather info about delhi" }
    OUTPUT: { "step": "OUTPUT": "content": "The cuurent weather in delhi is 20 C with some cloudy sky." }
        
    
"""

available_tools = {
    "run_command":run_command
}

class OutputFormat(BaseModel):
    step:str = Field(...,description="The id of the step for example START | PLAN | OUTPUT")
    content:Optional[str] = Field(None,description="The optional content for the step")
    tool:Optional[str] = Field(None,description="The id of the tool to call")
    input:Optional[str] = Field(None,description="The input params for the tool")


message_history = [{"role":"system","content":SYSTEM_PROMPT}]


while True:
    user_query = input("👉🏻 ")
    message_history.append({"role":"user","content":user_query})

    while True:

        response = client.chat.completions.parse(
            model="gemini-2.5-flash",
            response_format=OutputFormat,
            messages=message_history
        )

        raw_result = response.choices[0].message.content
        message_history.append({"role":"assistant","content":raw_result})

        parsed_result = response.choices[0].message.parsed

        if parsed_result.step == "START":
            print(f"😈 : {parsed_result.content}")
            continue

        if parsed_result.step == "TOOL":
            tool_to_call = parsed_result.tool
            tool_input = parsed_result.input
            print(f"Calling {tool_to_call} with input {tool_input}")
            tool_response = available_tools[tool_to_call](tool_input)
            message_history.append({"role":"developer","content":json.dumps(
                {"step":"OBSERVE","tool":tool_to_call,"input":tool_input,"output":tool_response}
            )})
            continue

        if parsed_result.step == "PLAN":
            print("🧠", parsed_result.content)
            continue

        if parsed_result.step == "OUTPUT":
            print(f"😈 : {parsed_result.content}")
            break;


print("\n\n\n\n")