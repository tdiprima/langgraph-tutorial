from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, TypedDict
import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

# Define the state: what data we'll track
class State(TypedDict):
    question: str          # The input question
    answer: str = ""       # The generated answer
    category: str = ""     # The category (e.g., "science")
    tags: list = []        # List of tags with weights

# Define structured outputs using Pydantic
class Answer(BaseModel):
    answer: str = Field(description="Readable answer to the question")

class Category(BaseModel):
    category: str = Field(description="Single word category, e.g., 'science'")

class Tags(BaseModel):
    tags: list = Field(description="List of dicts with 'tag' and 'weight' (0-1)")

# Set up the language model (we're using OpenAI's GPT-4o)
llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
)

# Node to generate an answer
def answer_node(state: State):
    prompt = f"Answer this question in a readable way: {state['question']}"
    response = llm.invoke(prompt).content
    return {"answer": response}

# Node to classify the question
def classify_node(state: State):
    prompt = f"Classify this question into a single word category: {state['question']}"
    response = llm.invoke(prompt).content
    return {"category": response}

# Node to generate tags
def tag_node(state: State):
    prompt = f"""Generate 4 tags for this question with weights (0-1) showing importance.
    Return ONLY valid JSON in this exact format:
    {{"tags": [
        {{"tag": "example_tag1", "weight": 0.9}},
        {{"tag": "example_tag2", "weight": 0.8}}
    ]}}
    
    Question: {state['question']}"""
    response = llm.invoke(prompt).content
    import json
    try:
        # Remove any leading/trailing whitespace that might affect JSON parsing
        cleaned_response = response.strip()
        tags = json.loads(cleaned_response)
        return tags  # The response should already be in {"tags": [...]} format
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {response}")
        # Return a default tags structure rather than failing
        return {"tags": [{"tag": "general", "weight": 1.0}]}

state = {"question": "What is photosynthesis?"}
print(answer_node(state))  # Outputs something like {"answer": "Photosynthesis is..."}
print(classify_node(state))  # Outputs {"category": "science"}
print(tag_node(state))  # Outputs {"tags": [{"tag": "plants", "weight": 0.9}, ...]}
