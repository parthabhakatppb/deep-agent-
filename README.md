# Deep Agents Chatbot

## 1. Project Overview
The **Deep Agents Chatbot** is an advanced AI-powered conversational assistant built using **Streamlit**, **LangChain**, and the **DeepAgents** framework. It acts as a comprehensive showcase of modern AI agent capabilities, including multi-agent orchestration, context engineering, tool usage (like web searching), and persistent memory across conversations.

This project was built to demonstrate proficiency with large language models (LLMs) and advanced agent frameworks, making it a perfect talking point for interviews related to AI/ML engineering, generative AI, and full-stack AI application development.

---

## 2. Directory Structure

```text
Deep-agents-With-Langchain-main/
│
├── main.py                  # A simple entry point script for testing
├── streamlit_app.py         # The core Streamlit application (main chatbot logic)
├── requirements.txt         # List of Python dependencies
├── pyproject.toml           # Modern Python project configuration
├── .env                     # Environment variables (API keys for OpenAI, Groq, Tavily)
└── deepagentsdemo/          # Directory containing agent memory, context, and skills
    ├── projects/
    │   └── AGENTS.md        # Persistent context file defining the agent's identity ("who you are")
    ├── skills/              # Markdown files defining specific skills the agent can learn and use
    └── *.ipynb              # Jupyter notebooks containing step-by-step feature demos
```

---

## 3. Detailed Code Breakdown

Below is a detailed, section-by-section explanation of the core files, specifically tailored to help you explain the project during an interview.

### 3.1 `streamlit_app.py` (The Core Engine)

This file is the heart of the application. It manages the UI, connects to the AI models, handles memory, and orchestrates the AI agents.

#### **Imports & Environment Setup**
```python
import os
import uuid
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load API keys from the .env file
ROOT_DIR = Path(__file__).parent
DEMO_DIR = ROOT_DIR / "deepagentsdemo"
load_dotenv(ROOT_DIR / ".env")
```
**Explanation:** 
- We import standard libraries (`os`, `uuid`, `pathlib`) for path management and generating unique thread IDs for conversations.
- `streamlit` is the frontend framework used to build the web UI.
- `load_dotenv` loads API keys (OpenAI, Groq, Tavily) from the `.env` file so they are securely injected into the environment.
- `pydantic` is used to strictly define the shape of the data the AI should return when we want structured outputs (like JSON).

#### **Framework Imports**
```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from tavily import TavilyClient
```
**Explanation:**
- `create_deep_agent`: The factory function from the `deepagents` library that builds the AI agent.
- `Backends (Filesystem, State, Store)`: These define where the agent stores its working memory (e.g., in RAM, on the hard drive, or in a persistent database).
- `MemorySaver` and `InMemoryStore`: LangGraph utilities to remember chat history (checkpointer) and long-term knowledge across different chat sessions.
- `TavilyClient`: A search engine built specifically for AI agents to retrieve live web data.

#### **Tool Creation (Web Search)**
```python
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def internet_search(query: str, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = False):
    """Run a web search"""
    return tavily_client.search(query, max_results=max_results, include_raw_content=include_raw_content, topic=topic)
```
**Explanation:** 
- We instantiate the Tavily client using our API key.
- The `internet_search` function is a **tool** that we give to the AI. If the user asks a question about recent events, the AI will automatically decide to execute this Python function, fetch web results, and read them before replying.

#### **Structured Output Definitions**
```python
class ResearchFindings(BaseModel):
    """Structured findings from a research task."""
    summary: str = Field(description="Summary of findings")
    confidence: float = Field(description="Confidence score from 0 to 1")
    sources: list[str] = Field(description="List of source URLs")
```
**Explanation:**
- We define a Pydantic model. When a specialized sub-agent (the "structured-researcher") performs research, it is forced to return a JSON object that exactly matches this structure (a summary, a confidence score, and a list of URLs). This is crucial for building robust APIs on top of LLMs.

#### **Context Engineering (Memory & Skills)**
```python
def load_agents_md() -> str:
    path = DEMO_DIR / "projects" / "AGENTS.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""

def load_skill_seed_files() -> dict:
    # Reads markdown files from deepagentsdemo/skills/ 
    # Converts them to virtual files for the AI to read.
```
**Explanation:**
- Context engineering is how we give the AI a long-term "brain". 
- `load_agents_md`: Reads a markdown file that contains the agent's core instructions and persona.
- `load_skill_seed_files`: Scans the `skills/` directory for markdown files. These files contain reusable instructions (e.g., "How to write Python code", "How to query AWS"). The AI can read these virtual files dynamically when it needs them.

#### **The Agent Factory (`build_agent`)**
```python
def build_agent(cfg: dict):
    # Selects Backend
    if cfg["backend"] == "StateBackend (in-state, per thread)":
        backend = StateBackend() ...
    elif cfg["backend"] == "FilesystemBackend (real disk)":
        backend = FilesystemBackend(...) ...
    else:
        backend = StoreBackend(...) ...
```
**Explanation:**
- This function builds the agent based on user settings from the UI.
- **StateBackend**: Temporary memory stored in RAM. It forgets files when the chat restarts.
- **FilesystemBackend**: The AI can literally write to your hard drive (confined to a safe directory).
- **StoreBackend**: LangGraph cross-thread memory. The AI remembers files across completely different conversation threads.

```python
    if cfg["use_subagents"]:
        subagents.append({
            "name": "research-agent",
            "tools": [internet_search],
            ...
        })
```
**Explanation:**
- This is the **Multi-Agent Orchestration** piece. We define specialized "sub-agents". The main agent acts as a manager; if it gets a complex research question, it delegates the task to the `research-agent`, which executes tools in an isolated context and reports back.

```python
    kwargs = dict(
        model=cfg["model"],
        tools=[internet_search],
        system_prompt=cfg["system_prompt"],
        backend=backend,
        checkpointer=st.session_state.checkpointer,
    )
    return create_deep_agent(**kwargs), seed_files
```
**Explanation:**
- Finally, it calls `create_deep_agent()` passing all the configurations (model, tools, system prompt, subagents, skills, memory backend).

#### **Streamlit UI and Event Loop**
```python
st.set_page_config(page_title="Deep Agents Chatbot", page_icon="🧠", layout="wide")
# Session State Initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
```
**Explanation:**
- `st.session_state` is used to persist variables (like chat history and memory stores) across UI refreshes. We generate a unique `thread_id` so the memory checkpointer knows which conversation it is currently dealing with.

```python
if prompt := st.chat_input("Ask me anything..."):
    # ...
    with st.spinner("🧠 Deep agent planning, researching, delegating…"):
        result = st.session_state.agent.invoke(payload, config=config)
```
**Explanation:**
- When the user types a prompt, we append it to the chat history and call `.invoke()` on our AI agent.
- LangChain/LangGraph processes the AI's internal loops (e.g., thinking -> deciding to search the web -> getting results -> synthesizing an answer). The `config` passes the `thread_id` so it remembers previous messages.

---

### 3.2 `pyproject.toml` & `requirements.txt`
Both files specify the dependencies needed to run the project.
- **langchain & langchain-openai/langchain-groq**: The core orchestration framework and adapters for different LLM providers (OpenAI GPT models, Groq open-source models).
- **deepagents**: A higher-level framework built on LangGraph that simplifies agent creation, memory backends, and sub-agent delegation.
- **streamlit**: The framework used for building the graphical user interface.
- **tavily-python**: The SDK for the Tavily web search API.

---

## 4. Interview Talking Points

If asked about this project in an interview, focus on these advanced AI concepts that you implemented:
1. **Agentic Workflows**: Unlike standard ChatGPT wrappers, this project implements a dynamic loop. The agent can reason, plan (using a `write_todos` internal tool), and decide when to use external APIs (web search).
2. **Multi-Agent Architecture**: You implemented a Manager-Worker dynamic. The main agent routes complex tasks to specialized sub-agents (e.g., `structured-researcher`), preventing the main agent from getting confused by too much context.
3. **Structured Outputs**: You enforced API reliability by using Pydantic schemas, forcing the LLM to return strictly typed JSON objects (Summary, Confidence, Sources) rather than raw text.
4. **Context Engineering & Retrieval**: Instead of stuffing the prompt with thousands of words, you created a virtual file system (`/skills/`). The agent can read these files on demand, simulating "learning" without fine-tuning.
5. **Persistent Memory**: By utilizing LangGraph's checkpointers and StoreBackend, you solved the "amnesia" problem of LLMs, allowing the agent to remember context across completely different conversation threads.

---

## 5. How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup API Keys:**
   Create a `.env` file in the root directory and add your keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   TAVILY_API_KEY=your_tavily_key
   ```
3. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```
