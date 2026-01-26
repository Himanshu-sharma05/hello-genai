import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from openai import OpenAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.checkpoint.mongodb import MongoDBSaver


load_dotenv()

client = OpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class State(TypedDict):
    messages:Annotated[list,add_messages]


def chat_bot(state:State):
    api_messages = []
    for msg in state["messages"]:
        if isinstance(msg,HumanMessage):
            api_messages.append({"role":"user","content":msg.content})
        elif isinstance(msg,AIMessage):
            api_messages.append({"role":"assistant","content":msg.content})
        elif isinstance(msg,SystemMessage):
            api_messages.append({"role":"system","content":msg.content})

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=api_messages
    )
    result = response.choices[0].message.content
    return {"messages":[result]}

graph_builder = StateGraph(State)

graph_builder.add_node("chat_bot",chat_bot)

graph_builder.add_edge(START,"chat_bot")
graph_builder.add_edge("chat_bot",END)

graph = graph_builder.compile()

def graph_builder_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

DB_URI = "mongodb://admin:admin@localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph_with_checkpointer = graph_builder_with_checkpointer(checkpointer=checkpointer)

    config = {
            "configurable": {
                "thread_id": "Hisenberg" # user_id
            }
        }
    for chunk in graph_with_checkpointer.stream(
        State({"messages":["Now say my name"]}),
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
        
