from dotenv import load_dotenv
from openai import OpenAI
import os 
from pydantic import BaseModel,Field
from typing import Optional

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

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
        START: 
        
    
"""