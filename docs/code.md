## question\_processor.py


### 1. Imports and Setup

```python
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, TypedDict
import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
import json
```

**What we're doing**: We're gathering all the tools we need, like picking up weapons and maps before a game level.

**Why we're doing it**:

- `load_dotenv()` loads secret stuff (like API keys) from a `.env` file so the code can talk to Azure's AI without exposing passwords.
- `TypedDict`, `BaseModel`, etc., are for organizing data neatly (like labeling inventory items).
- `langgraph` and `AzureChatOpenAI` are the main engines: LangGraph builds the flowchart, and Azure AI answers questions.
- `json` helps format the final output so it's readable for humans or computers.

---

### 2. Defining the State (Shared Notebook)

```python
class State(TypedDict):
    question: str
    answer: str = ""
    category: str = ""
    tags: list = []
```

**What we're doing**: We're creating a "State" class, which is like a shared quest log that everyone updates as they work.

**Why we're doing it**:

- It holds the question (what we're starting with), the answer (what we'll find), a category (like a folder label), and tags (keywords with importance).
- Keeps everything organized so each part of the process knows what's going on. It's the central hub for data.

---

### 3. Defining Output Structures (Data Templates)

```python
class Answer(BaseModel):
    answer: str = Field(description="Readable answer to the question")

class Category(BaseModel):
    category: str = Field(description="Single word category, e.g., 'science'")

class Tags(BaseModel):
    tags: list = Field(description="List of dicts with 'tag' and 'weight' (0-1)")
```

**What we're doing**: We're making templates (like forms) that tell the AI what kind of output we want.

**Why we're doing it**:

- These ensure the AI gives us structured data (e.g., an answer as a string, a category as one word, tags as a list with weights).
- Pydantic (`BaseModel`) enforces rules, so we don't get messy or wrong answers.

---

### 4. Setting Up the AI Model

```python
llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
)
```

**What we're doing**: We're connecting to Azure's AI (like hiring a smart NPC).

**Why we're doing it**:

- This is the brain that will answer questions, classify them, and generate tags. It uses credentials from the `.env` file to log in.
- We need it to do the heavy lifting of understanding language and generating responses.

---

### 5. Node Functions (The Task Workers)
Each of these functions is a "node" that does one job in the pipeline.

#### a. `node_answer`

```python
def node_answer(state: State):
    prompt = f"Answer this question in a readable way: {state['question']}"
    response = llm.invoke(prompt).content
    return {"answer": response}
```

**What we're doing**: We ask the AI to answer the question in simple terms.

**Why we're doing it**:

- This is the first step: get a clear answer. We pass the question from the State, the AI responds, and we store it back in State.

#### b. `node_classify`

```python
def node_classify(state: State):
    prompt = f"Classify this question into a single word category: {state['question']}"
    response = llm.invoke(prompt).content
    return {"category": response}
```

**What we're doing**: We ask the AI to label the question (e.g., "science," "history").

**Why we're doing it**:

- Helps organize the question. Knowing the category makes it easier to file or search later.

#### c. `node_tag`

```python
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
```

**What we're doing**: We ask the AI to create 4 keywords (tags) and rate their importance (0-1).

**Why we're doing it**:

- Tags help summarize and prioritize info. The JSON format ensures the output is clean and usable.
- The `try/except` handles errors in case the AI messes up the JSON, defaulting to a safe "general" tag.

---

### 6. Building the Graph (Flowchart)

```python
graph = StateGraph(State)

graph.add_node("node_answer", node_answer)
graph.add_node("node_classify", node_classify)
graph.add_node("node_tag", node_tag)
graph.add_node("dispatch", dispatch_node)
graph.add_node("combine", lambda x: x)  # Add combine node first

graph.set_entry_point("dispatch")
graph.add_edge("dispatch", "node_answer")
graph.add_edge("node_answer", "node_classify")
graph.add_edge("node_classify", "node_tag")
graph.add_edge("node_tag", "combine")

graph.set_finish_point("combine")

app = graph.compile()
```

**What we're doing**: We're building a flowchart (graph) that connects all the nodes in order.

**Why we're doing it**:

- LangGraph lets us create a sequence: start with "dispatch" (just passes the State), then answer, classify, tag, and finally combine.
- `set_entry_point` and `set_finish_point` define where the process starts and ends.
- `compile()` turns it into a runnable app.

---

### 7. Updating the Combine Node

```python
def combine_node(state: State):
    return {
        "question": state["question"],
        "answer": state["answer"],
        "category": state["category"],
        "tags": state["tags"]
    }

graph.nodes.pop("combine")  # Remove existing node
graph.add_node("combine", combine_node)
app = graph.compile()  # Recompile after updating
```

**What we're doing**: We're fixing the "combine" node to package all results neatly.

**Why we're doing it**:

- The old "combine" just passed data through. Now, it gathers everything (question, answer, category, tags) into one structure.
- We update the graph and recompile so the new logic works.

---

### 8. Running and Printing the Result

```python
result = app.invoke({"question": "What is photosynthesis?"})
import json
print(json.dumps(result, indent=2))
```

**What we're doing**: We run the whole pipeline with a test question and print the output as pretty JSON.

**Why we're doing it**:

- Tests the system. "What is photosynthesis?" is the input, and we want to see the full output (answer, category, tags).
- `json.dumps` makes the output readable for humans.

---

### Wrap-Up:
Each section is a step in a quest to turn a question into organized info. We use tools (imports), plan (State and models), delegate tasks (nodes), map the flow (graph), and deliver (combine and print). If you lose focus, just remember: it's a question → answer pipeline with helpers!

<br>
