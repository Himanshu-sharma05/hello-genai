import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Optional,Literal
from langgraph.graph import StateGraph,START,END
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class State(TypedDict):
    user_query:str
    llm_output:Optional[str]
    is_good:Optional[bool]


def chat_bot(state:State):
    print("Inside the chatbot node",state)
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role":"user","content":state.get("user_query")}]
    )
    result = response.choices[0].message.content
    state["llm_output"] = result
    return state

def conditional_node(state:State) -> Literal["end_node","gemini_node"]:
    print("inside the conditional node")
    if False:
        return "end_node"
    else:
        return "gemini_node"
    
def gemini_node(state:State):
    print("Inside the gemini Node")
    response = client.chat.completions.create(
        model="gemini-3-flash-preview",
        messages=[{"role":"user","content":state.get("user_query")}]
    )
    result = response.choices[0].message.content
    state["llm_output"] = "This is an updated Result: " + result
    return state

def end_node(state:State):
    print("inside the end node")
    return state


graph_builder = StateGraph(State)

graph_builder.add_node("chat_bot",chat_bot)
graph_builder.add_node("gemini_node",gemini_node)
graph_builder.add_node("end_node",end_node)


graph_builder.add_edge(START,"chat_bot")
graph_builder.add_conditional_edges("chat_bot",conditional_node)
graph_builder.add_edge("gemini_node","end_node")
graph_builder.add_edge("end_node",END)

graph = graph_builder.compile()

updated_state = graph.invoke(State({"user_query":"Hey im himanshu"}))
print(updated_state)