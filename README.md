# LangSmith Masterclass

This repository contains a comprehensive collection of LangChain examples demonstrating various patterns and best practices for building LLM applications. The examples progress from simple LLM calls to complex multi-step workflows with tracing, RAG (Retrieval-Augmented Generation), agents, and LangGraph implementations.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [File Descriptions](#file-descriptions)
  - [1. Simple LLM Call](#1-simple-llm-call)
  - [2. Sequential Chain](#2-sequential-chain)
  - [3. RAG Implementations](#3-rag-implementations)
  - [4. Agent with Tools](#4-agent-with-tools)
  - [5. LangGraph Workflow](#5-langgraph-workflow)
- [Environment Variables](#environment-variables)
- [Usage](#usage)

## 🎯 Overview

This masterclass covers:

- **Basic LLM Interactions**: Simple prompt → model → parser chains
- **Sequential Chains**: Multi-step processing with different models
- **RAG (Retrieval-Augmented Generation)**: Document-based Q&A systems with progressive improvements
- **Agents**: LLM agents with tool usage (web search, weather API)
- **LangGraph**: Complex stateful workflows for parallel processing

All examples are designed to work with **LangSmith** for tracing and monitoring LLM applications.

## 🔧 Prerequisites

- Python 3.8+
- OpenAI API key
- LangSmith account (for tracing - optional but recommended)

## ⚙️ Setup

1. **Clone or navigate to this directory**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the root directory with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=your_project_name
   ```

## 📁 File Descriptions

### 1. Simple LLM Call (`1_simple_llm_call.py`)

**What it does**: Demonstrates the most basic LangChain pattern - a simple prompt template, LLM model, and output parser chained together.

**Key Concepts**:
- `PromptTemplate`: Creates reusable prompt templates
- `ChatOpenAI`: OpenAI chat model wrapper
- `StrOutputParser`: Parses LLM output to string
- Chain composition using the pipe operator (`|`)

**Example**: Asks "What is the capital of Peru?" and prints the response.

**Run**:
```bash
python 1_simple_llm_call.py
```

---

### 2. Sequential Chain (`2_sequential_chain.py`)

**What it does**: Shows how to chain multiple LLM calls sequentially, where the output of one becomes input to the next. Uses different models for different tasks.

**Key Concepts**:
- Multi-step chains: `prompt1 → model1 → parser → prompt2 → model2 → parser`
- Using different models for different tasks (GPT-4o-mini for generation, GPT-4o for summarization)
- LangSmith configuration with run names, tags, and metadata

**Workflow**:
1. First model generates a detailed report on a topic
2. Second model creates a 5-point summary from that report

**Example**: Generates a report on "Unemployment in India" and then summarizes it.

**Run**:
```bash
python 2_sequential_chain.py
```

---

### 3. RAG Implementations

This section contains four progressively improved versions of a RAG (Retrieval-Augmented Generation) system that answers questions based on a PDF document (`islr.pdf`).

#### 3.1. RAG v1 (`3_rag_v1.py`)

**What it does**: Basic RAG implementation without LangSmith tracing.

**Key Concepts**:
- `PyPDFLoader`: Loads PDF documents
- `RecursiveCharacterTextSplitter`: Splits documents into chunks
- `OpenAIEmbeddings`: Creates vector embeddings
- `FAISS`: Vector database for similarity search
- `RunnableParallel`: Runs retrieval and question processing in parallel

**Workflow**:
1. Load PDF → Split into chunks → Create embeddings → Build vector store
2. For each question: Retrieve relevant chunks → Format context → Generate answer

**Run**:
```bash
python 3_rag_v1.py
```

#### 3.2. RAG v2 (`3_rag_v2.py`)

**What it does**: Adds LangSmith tracing to the RAG pipeline using the `@traceable` decorator.

**Key Concepts**:
- `@traceable` decorator: Wraps functions to create traceable spans in LangSmith
- Hierarchical tracing: Setup functions are traced separately from query execution
- Named traces: Each function gets a descriptive name for better observability

**Improvements over v1**:
- Traces PDF loading, document splitting, and vector store building
- Separate traces for setup vs. query execution
- Better visibility into each step of the pipeline

**Run**:
```bash
python 3_rag_v2.py
```

#### 3.3. RAG v3 (`3_rag_v3.py`)

**What it does**: Organizes tracing with a parent-child hierarchy, grouping setup steps under a single parent trace.

**Key Concepts**:
- Parent-child trace relationships
- Top-level root trace that encompasses the entire operation
- Tags for categorizing traces (e.g., `["setup"]`)

**Improvements over v2**:
- All setup steps (load, split, build) are children of a parent "setup_pipeline" trace
- The entire operation (setup + query) is under a root "pdf_rag_full_run" trace
- Better trace organization in LangSmith UI

**Run**:
```bash
python 3_rag_v3.py
```

#### 3.4. RAG v4 (`3_rag_v4.py`)

**What it does**: Adds intelligent caching and index persistence to avoid rebuilding the vector store on every run.

**Key Concepts**:
- File fingerprinting: Uses SHA256 hash to detect PDF changes
- Index caching: Saves and loads FAISS indices from disk
- Cache invalidation: Rebuilds only when PDF or parameters change
- Metadata tracking: Stores configuration in `meta.json`

**Improvements over v3**:
- **Performance**: Skips expensive embedding and indexing if PDF hasn't changed
- **Persistence**: Saves indices to `.indices/` directory
- **Smart caching**: Detects changes in PDF, chunk size, overlap, or embedding model
- **Force rebuild option**: Can force rebuild even if cache exists

**Workflow**:
1. Calculate cache key based on PDF fingerprint + parameters
2. If cache exists → load from disk
3. If cache missing or invalid → build new index and save
4. Query using cached or new index

**Run**:
```bash
python 3_rag_v4.py
```

---

### 4. Agent with Tools (`4_agent_copy.py`)

**What it does**: Demonstrates an LLM agent that can use external tools to answer questions. The agent decides when and how to use tools based on the query.

**Key Concepts**:
- `@tool` decorator: Converts functions into LangChain tools
- `create_react_agent`: Creates a ReAct (Reasoning + Acting) agent
- `AgentExecutor`: Executes the agent with tool access
- Tool integration: Agent can call multiple tools in sequence

**Available Tools**:
1. **DuckDuckGo Search**: Performs web searches and returns top 3 results
2. **Weather API**: Fetches current weather data for any city using Weatherstack API

**Example Queries**:
- "What is the current temperature in Hyderabad?"
- "What is the release date of Dhadak 2?" (requires web search)

**Workflow**:
1. Agent receives query
2. Agent decides if tools are needed
3. Agent calls appropriate tool(s)
4. Agent synthesizes final answer from tool results

**Run**:
```bash
python 4_agent_copy.py
```

---

### 5. LangGraph Workflow (`5_langgraph.py`)

**What it does**: Implements a complex stateful workflow using LangGraph for parallel essay evaluation across multiple dimensions.

**Key Concepts**:
- `StateGraph`: LangGraph's graph-based workflow builder
- `TypedDict`: Defines the state structure
- Parallel execution: Multiple evaluation nodes run simultaneously
- Structured output: Uses Pydantic models for consistent LLM responses
- State aggregation: Combines results from parallel nodes

**Workflow Structure**:
```
START
  ├─→ evaluate_language ──┐
  ├─→ evaluate_analysis ──┼─→ final_evaluation → END
  └─→ evaluate_thought ────┘
```

**Evaluation Dimensions**:
1. **Language Quality**: Grammar, vocabulary, writing style
2. **Depth of Analysis**: Critical thinking, argumentation
3. **Clarity of Thought**: Structure, coherence, logical flow

**Features**:
- Each dimension is evaluated in parallel (fan-out pattern)
- All results are aggregated into a final evaluation
- Uses structured output (Pydantic) for consistent scoring
- Comprehensive tracing with tags and metadata

**Example**: Evaluates an essay about "India and AI Time" across all three dimensions and provides:
- Individual feedback for each dimension
- Individual scores (0-10) for each dimension
- Overall feedback summary
- Average score

**Run**:
```bash
python 5_langgraph.py
```

---

## 🔐 Environment Variables

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=sk-...

# Optional (for LangSmith tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=langsmith-masterclass
```

**Note**: Some examples set `LANGCHAIN_PROJECT` programmatically, but you can override it in `.env`.

---

## 🚀 Usage

### Running Individual Examples

Each file can be run independently:

```bash
# Basic examples
python 1_simple_llm_call.py
python 2_sequential_chain.py

# RAG examples (requires islr.pdf)
python 3_rag_v1.py
python 3_rag_v2.py
python 3_rag_v3.py
python 3_rag_v4.py

# Agent example
python 4_agent_copy.py

# LangGraph example
python 5_langgraph.py
```

### Interactive Examples

Some examples are interactive and will prompt for input:
- **RAG examples**: Ask questions about the PDF
- **Agent example**: Enter queries that may require tool usage

### PDF Requirement

The RAG examples (`3_rag_v*.py`) require a PDF file named `islr.pdf` in the same directory. You can:
1. Place your own PDF file and rename it to `islr.pdf`, or
2. Modify the `PDF_PATH` variable in each RAG script to point to your PDF

---

## 📊 Learning Path

Recommended order for learning:

1. **Start Simple**: `1_simple_llm_call.py` - Understand basic chains
2. **Sequential Processing**: `2_sequential_chain.py` - Learn multi-step chains
3. **RAG Basics**: `3_rag_v1.py` - Understand retrieval-augmented generation
4. **Add Tracing**: `3_rag_v2.py` - Learn LangSmith tracing
5. **Organize Traces**: `3_rag_v3.py` - Understand trace hierarchies
6. **Optimize**: `3_rag_v4.py` - Learn caching and persistence
7. **Agents**: `4_agent_copy.py` - Build agents with tools
8. **Complex Workflows**: `5_langgraph.py` - Master stateful workflows

---

## 🛠️ Technologies Used

- **LangChain**: Framework for building LLM applications
- **LangSmith**: Observability and tracing platform
- **OpenAI**: GPT models (GPT-4o, GPT-4o-mini)
- **FAISS**: Vector similarity search
- **LangGraph**: Stateful workflow orchestration
- **Pydantic**: Data validation and structured output
- **DuckDuckGo Search**: Web search tool
- **Weatherstack API**: Weather data tool

---

## 📝 Notes

- All examples use OpenAI models. Ensure you have sufficient API credits.
- LangSmith tracing is optional but highly recommended for debugging and monitoring.
- The RAG examples use `islr.pdf` by default - replace with your own PDF or update the path.
- The agent example includes a hardcoded Weatherstack API key - consider using environment variables for production.
- Index caching in RAG v4 saves to `.indices/` directory - you can delete this to force rebuild.

---

## 🤝 Contributing

This is a masterclass repository. Feel free to experiment, modify, and extend the examples for your learning!

---

## 📄 License

This repository is for educational purposes as part of the LangSmith masterclass.

