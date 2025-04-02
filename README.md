## Question Processing with LangGraph

This repository contains a single Python script that uses LangGraph and AzureChatOpenAI to process questions. It answers questions, classifies them into categories, and generates weighted tags.

### What It Does
- Takes a question (e.g., "What is photosynthesis?").
- Generates a readable answer, a single-word category, and 4 tags with importance weights (0-1).
- Outputs results in JSON format.

### How to Use
1. Install dependencies: `pip install langgraph langchain-openai pydantic python-dotenv`.
2. Set up a `.env` file with Azure OpenAI credentials (`AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_VERSION`).
3. Run the script: `python question-processor.py`.

### File
- `question-processor.py`: The main script.

Enjoy automated question processing!
