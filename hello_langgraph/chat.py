from dotenv import load_dotenv
import os
from openai import OpenAI
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
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


def sample_node(state:State):
    print("\n\n Inside the Sample Node",state)
    return {"messages":["This is appended by the sample node."]}

graph_builder = StateGraph(State)

graph_builder.add_node("chat_bot",chat_bot)
graph_builder.add_node("sample_node",sample_node)

graph_builder.add_edge(START,"chat_bot")
graph_builder.add_edge("chat_bot","sample_node")
graph_builder.add_edge("sample_node",END)

graph = graph_builder.compile()

updatd_state = graph.invoke(State({"messages":["Hey my name is Himanshu"]}))

print("\n\nupdated state",updatd_state)