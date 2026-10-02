"""ReAct research agent that chooses search vs calculation at each step."""

from __future__ import annotations

import os
from typing import Any

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from tools.calculator import calculator as calculator_tool
from tools.search_tool import web_search_tool
from tools.wikipedia_tool import wikipedia_tool

REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat NYeh `OpenAIAuthenticationError` tabhi aata hai jab Python code OpenAI ke direct endpoint (`api.openai.com`) par request bhej raha ho, OpenRouter ke URL par nahi.

Aapke `research_agent.py` file me **`_gemini_chat_model`** function abhi bhi puraani configuration use kar raha hai (`base_url` ke naam se, jo LangChain ignore kar raha hai).

Isay completely fix karne ke liye:

1. GitHub par **[`agents/research_agent.py`](https://github.com/muaza9241-cloud/research-agent/edit/main/agents/research_agent.py)** edit link par jayein.

2. Lines 46-53 wale function ko is exact code se replace kar dein:

```python
def _gemini_chat_model(api_key: str) -> BaseChatModel:
    return ChatOpenAI(
        model_name="google/gemini-2.5-flash",
        openai_api_key=api_key,
        openai_api_base="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)",
        max_tokens=2000,
        temperature=0
    )
