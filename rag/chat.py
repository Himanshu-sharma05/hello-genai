from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_database = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)

user_query = input("> Enter your query: ")

search_results = vector_database.similarity_search(query=user_query)

context = "\n\n\n".join([f"Page Content:{result.page_content}\n Page Number : {result.metadata['page_label']}\n File Location : {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who answeres user query based on the available context retrieved from a PDF file along with page_contents and page number.
    You don't only provide information of the source but also try to answer user query on based on the context like a human.

 You should only ans the user based on the following context and navigate the user to open the right page number to know more.

 Context:
 {context}
"""

print(context)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_query}]
)

print(f"😸: {response.choices[0].message.content}")