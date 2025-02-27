---
title: "Understanding AI Agents (Code Agents) - Huggingface Course"
excerpt: "Code agents are a type of AI agent that generate and execute Python code to perform actions. Instead of relying on structured JSON outputs or predefined workflows, code agents allow Large Language Models (LLMs) to dynamically write, execute, and refine code as part of their decision-making process"
collection: teaching
colab_url: "https://colab.research.google.com/drive/1abTdaGJRl3JoL7XB1JxRSf4ZsIcrbW5n"
github_url: "https://github.com/japhari/natural-language-processing/blob/main/code_agents.ipynb"
thumbnail: "/images/publication/agentic.png"
type: "Natural Language Processing"
permalink: /teaching/2025/Understanding-agents
venue: "PTIT , Department of Computer Science"
date: 2025-02-27
location: "Hanoi, Vietnam"
categories:
  - teaching
tags:
  - nlp
  - agents
---

### Credit - Huggingface Agents Course

## Introduction

Code agents are a type of AI agent that generate and execute Python code to perform actions. Instead of relying on structured JSON outputs or predefined workflows, code agents allow Large Language Models (LLMs) to dynamically write, execute, and refine code as part of their decision-making process

An agentic framework is not always necessary when building applications with Large Language Models (LLMs). While they offer flexibility in workflows for efficiently solving complex tasks, there are scenarios where predefined workflows are sufficient.

If the agent's design is straightforward, such as a simple chain of prompts, using plain code ensures full control and system transparency without unnecessary abstractions. However, as workflows become more intricate—such as allowing an LLM to call functions or employing multiple agents—these frameworks become beneficial.

### Essential Components of an Agentic Framework

To manage agent-driven workflows effectively, certain core features are required:

- **LLM Engine**: The backbone of the system.
- **Accessible Tools**: A collection of tools that the agent can use.
- **Parsing Mechanisms**: Extract tool calls from LLM output.
- **System Prompt**: Align with parser specifications.
- **Memory System**: Store and recall previous interactions.
- **Error Handling**: Log errors and implement retry strategies to correct LLM mistakes.

### Agentic Frameworks Overview

The following are notable frameworks for developing agents:

| Framework      | Description                                   | Unit Author         |
| -------------- | --------------------------------------------- | ------------------- |
| **smolagents** | A lightweight agent framework by Hugging Face | Sergio Paniego - HF |

### Introduction to smolagents

Welcome to this module, where we will explore smolagents, a simple yet powerful framework designed for AI agent development. smolagents allows LLMs to interact with real-world applications by searching, generating content, and executing actions.

### Why Use smolagents?

smolagents offers several advantages:

- **Simplicity**: Minimal abstractions, making it easy to understand and extend.
- **Flexible LLM Support**: Compatible with Hugging Face tools and external APIs.
- **Code-First Approach**: Generates and executes code directly, avoiding JSON parsing complexities.
- **HF Hub Integration**: Enables tool sharing and collaborative development.

### When to Use smolagents

Consider smolagents when:

- You need a lightweight, minimalistic solution.
- Rapid experimentation is required without complex configurations.
- The application logic is straightforward.

### Code vs. JSON Actions

Unlike other frameworks that use JSON for action specification, smolagents generates executable code. This eliminates the need for JSON parsing, ensuring direct execution and higher efficiency.

## Key Components in smolagents

### CodeAgents

CodeAgents are a core feature of smolagents. Instead of JSON or text outputs, these agents generate Python code snippets for execution.

### ToolCallingAgents

ToolCallingAgents generate JSON/text tool calls that require parsing before execution. This module covers their implementation and key differences from CodeAgents.

### Tools

Tools act as functional components in agentic systems. They are implemented using the `@tool` decorator or `Tool` class.

### Retrieval Agents

Retrieval Agents allow LLMs to search and retrieve relevant information from various sources using vector stores and RAG (Retrieval-Augmented Generation) techniques.

### Multi-Agent Systems

Combining multiple agents enhances capabilities, such as integrating web search agents with code execution agents to build robust AI-driven applications.

### Vision and Browser Agents

Vision agents extend AI capabilities by incorporating Vision-Language Models (VLMs), enabling image analysis and web browsing functionalities.

## Model Integration in smolagents

smolagents supports various LLM integrations:

- **TransformersModel**: Uses local transformers pipeline.
- **HfApiModel**: Connects to Hugging Face’s Serverless Inference API.
- **LiteLLMModel**: Lightweight interaction with models.
- **OpenAIServerModel**: Connects to OpenAI-compatible services.
- **AzureOpenAIServerModel**: Integration with Azure OpenAI deployments.

This flexibility allows for seamless adaptation based on project requirements.

## Building Effective Agents with smolagents

### Example 1: Searching for a Party Playlist

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
agent.run("Search for the best music recommendations for a party at Wayne's mansion.")
```

### Example 2: Custom Tool for Menu Planning

```python
from smolagents import CodeAgent, tool, HfApiModel

@tool
def suggest_menu(occasion: str) -> str:
    menus = {
        "casual": "Pizza, snacks, and drinks.",
        "formal": "3-course dinner with wine and dessert.",
        "superhero": "Buffet with high-energy and healthy food."
    }
    return menus.get(occasion, "Custom menu for the butler.")

agent = CodeAgent(tools=[suggest_menu], model=HfApiModel())
agent.run("Prepare a formal menu for the party.")
```

### Example 3: Calculating Party Preparation Time

```python
from smolagents import CodeAgent, HfApiModel
import datetime

agent = CodeAgent(tools=[], model=HfApiModel(), additional_authorized_imports=['datetime'])

agent.run("Calculate when the party will be ready if we start now.")
```

## Sharing Agents on the Hugging Face Hub

smolagents allows users to share their agents with the community. To upload an agent:

```python
agent.push_to_hub('username/AlfredAgent')
```

To retrieve the agent:

```python
alfred_agent = agent.from_hub('username/AlfredAgent')
alfred_agent.run("Give me the best playlist for a party at Wayne's mansion.")
```

## Debugging with OpenTelemetry and Langfuse

To monitor agent performance, integrate OpenTelemetry with Langfuse:

## Setup

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
```

## Configuration

```python
import os
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
```

## Enabling Instrumentation

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
SmolagentsInstrumentor().instrument()
```

## **Code Agents**

Code agents are a core component of the **smolagents** framework, allowing AI models to execute Python code directly instead of relying on predefined workflows or JSON-based tool calls.

Unlike traditional methods that use JSON outputs requiring parsing, **code agents write and execute Python code directly**. This approach simplifies execution and allows more flexibility in agent behavior.

### **Advantages of Code Agents**

1. **Direct Execution** – No need to parse JSON before execution.
2. **Better Code Reuse** – Allows the integration of existing Python functions.
3. **Handles Complex Data** – Works well with images, structured data, and APIs.
4. **Natural for LLMs** – AI models are trained on Python, making this approach more efficient.

---

### **How Does a Code Agent Work?**

The execution of a code agent follows a sequence of steps:

**Step 1: Setting Up the Code Agent**

- Initialize the **CodeAgent** and specify a model.
- Define any tools the agent may need.

**Step 2: Processing User Input**

- The agent stores the **initial user query**.
- Past interactions are saved in memory.

**Step 3: Generating Code**

- The agent constructs **Python tool calls** based on the query.
- The generated code is extracted and executed.

**Step 4: Executing and Logging**

- The results of the executed Python code are logged.
- If errors occur, they are managed through built-in retry mechanisms.

---

## **Examples of Code Agents in Action**

**Example 1: Fetching Weather Information**

A code agent can be used to fetch weather details from an API.

**Step 1: Define a Custom Weather Tool**

```python
from smolagents import CodeAgent, tool, HfApiModel
import requests

@tool
def get_weather(city: str) -> str:
    """
    Fetches weather information for a given city.
    Args:
        city: Name of the city.
    """
    response = requests.get(f"https://wttr.in/{city}?format=3")
    return response.text

agent = CodeAgent(tools=[get_weather], model=HfApiModel())
```

**Step 2: Run the Agent**

```python
agent.run("Get the current weather in New York.")
```

- The agent calls the `get_weather` function.
- It retrieves the live weather data and returns it.

---

### **Example 2: Performing a Math Calculation**

A code agent can solve mathematical problems by executing Python code.

**Step 1: Define a Calculation Tool**

```python
@tool
def solve_equation(equation: str) -> str:
    """
    Solves a given mathematical equation.
    Args:
        equation: A string representation of the equation (e.g., '2 + 2 * 3').
    """
    try:
        result = eval(equation)
        return f"The result of {equation} is {result}."
    except Exception as e:
        return f"Error solving equation: {e}"
```

**Step 2: Run the Agent**

```python
agent = CodeAgent(tools=[solve_equation], model=HfApiModel())
agent.run("Solve the equation: 5 * (10 + 2)")
```

- The agent evaluates the expression and returns the answer.

---

### **Example 3: Retrieving Stock Prices**

A code agent can retrieve live stock market data.

**Step 1: Define a Stock Price Tool**

```python
import yfinance as yf

@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetches the latest stock price for a given stock symbol.
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL' for Apple).
    """
    stock = yf.Ticker(symbol)
    price = stock.history(period="1d")["Close"].iloc[-1]
    return f"The latest price for {symbol} is ${price:.2f}."
```

**Step 2: Run the Agent**

```python
agent = CodeAgent(tools=[get_stock_price], model=HfApiModel())
agent.run("Get the stock price of Tesla (TSLA).")
```

- The agent calls `get_stock_price`, retrieves market data, and returns the latest price.

---

## **How to Enhance Code Agents**

### **Adding Python Imports**

By default, code agents operate within a secure execution environment. Some imports may need explicit authorization.

**Example: Using NumPy for Scientific Computation**

```python
agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    additional_authorized_imports=['numpy']
)

agent.run("Compute the square root of 144 using NumPy.")
```

- The agent now has access to NumPy for calculations.

---

## **Debugging Code Agents**

Agents can sometimes return errors. To track execution, **OpenTelemetry** and **Langfuse** can be integrated.

**Step 1: Install Dependencies**

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
```

**Step 2: Set Up API Keys**

```python
import os
import base64

LANGFUSE_PUBLIC_KEY="your_public_key"
LANGFUSE_SECRET_KEY="your_secret_key"
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
```

**Step 3: Enable Logging**

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
```

- This ensures each agent run is logged for debugging.

---

## **Sharing a Code Agent**

Once an agent is built, it can be **shared on the Hugging Face Hub**.

**Step 1: Upload the Agent**

```python
agent.push_to_hub('your_username/CustomAgent')
```

**Step 2: Download and Use the Agent**

```python
agent = CodeAgent.from_hub('your_username/CustomAgent')
agent.run("Fetch the latest news headlines.")
```

---

---
