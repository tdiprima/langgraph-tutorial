from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, TypedDict
import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
import json

class State(TypedDict):
    question: str
    answer: str = ""
    category: str = ""
    tags: list = []

class Answer(BaseModel):
    answer: str = Field(description="Readable answer to the question")

class Category(BaseModel):
    category: str = Field(description="Single word category, e.g., 'science'")

class Tags(BaseModel):
    tags: list = Field(description="List of dicts with 'tag' and 'weight' (0-1)")

llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
)

def node_answer(state: State):
    prompt = f"Answer this question in a readable way: {state['question']}"
    response = llm.invoke(prompt).content
    return {"answer": response}

def node_classify(state: State):
    prompt = f"Classify this question into a single word category: {state['question']}"
    response = llm.invoke(prompt).content
    return {"category": response}

def node_tag(state: State):
    prompt = f"""Generate 4 tags for this question with weights (0-1) showing importance.
    Return ONLY valid JSON in this exact format, and do not add any additional text or "```" characters:
    {{"tags": [
        {{"tag": "example_tag1", "weight": 0.9}},
        {{"tag": "example_tag2", "weight": 0.8}}
    ]}}
    
    Question: {state['question']}"""
    response = llm.invoke(prompt).content
    try:
        cleaned_response = response.strip()
        tags = json.loads(cleaned_response)
        return tags
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {response}")
        return {"tags": [{"tag": "general", "weight": 1.0}]}

def dispatch_node(state: State):
    return state  # Pass the state unchanged

# state = {"question": "What is photosynthesis?"}
# print(node_answer(state))
# print(node_classify(state))
# print(node_tag(state))

graph = StateGraph(State)

graph.add_node("node_answer", node_answer)
graph.add_node("node_classify", node_classify)
graph.add_node("node_tag", node_tag)
graph.add_node("dispatch", dispatch_node)
graph.add_node("combine", lambda x: x)

graph.set_entry_point("dispatch")
graph.add_edge("dispatch", "node_answer")
graph.add_edge("node_answer", "node_classify")
graph.add_edge("node_classify", "node_tag")
graph.add_edge("node_tag", "combine")

graph.set_finish_point("combine")

app = graph.compile()

result = app.invoke({"question": "What is photosynthesis?"})
print(result)
