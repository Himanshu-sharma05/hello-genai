import os
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory
import json

load_dotenv()

client = OpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

config = {
    "version":"v1.1",
    "embedder":{
        "provider":"gemini",
        "config":{
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": "models/gemini-embedding-001",
            "output_dimensionality": 1536
        }},
    "llm":{
        "provider":"gemini",
        "config":{"api_key":os.getenv("GEMINI_API_KEY"),"model":"gemini-2.5-flash"}
    },
    "vector_store":{
        "provider":"qdrant",
        "config":{"host":"localhost","port":6333},
    }
    
}

mem_client = Memory.from_config(config)


while True:

    user_query = input(">")
    search_memory = mem_client.search(query=user_query,user_id="himanshu")

    memories = [f"ID: {mem.get("id")}\n Memory:{mem.get("memory")}" for mem in search_memory.get("results")]

    print("Found Memories",memories)

    SYSTEM_PROMPT = f"""
        Here is the context about the user:
        {json.dumps(memories)}
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_query}]
    )
    
    result = response.choices[0].message.content

    print("AI: ",result)

    mem_client.add(
        user_id="himanshu",
        messages=[{"role":"user","content":user_query},{"role":"assistant","content":result}]
    )
    print("Memory Saved ...")

