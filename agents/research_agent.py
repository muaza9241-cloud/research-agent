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
    """You are a careful research agent. You decide dynamically whether to
search the web or calculate numbers. Prefer live search for prices, places,
schedules, and current facts. Use the calculator for totals, remaining budget,
per-day splits, and other arithmetic. Always cite source URLs from search
results in the Final Answer.

You have access to these tools:

{tools}

Use this exact format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the complete answer, including sources (URLs) when search was used

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

def _gemini_chat_model(api_key: str) -> BaseChatModel:
    return ChatOpenAI(
        model_name="google/gemini-2.5-flash",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=2000,
        temperature=0
    )

def _load_llm() -> BaseChatModel:
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY missing in .env file")

    return ChatOpenAI(
        model_name="meta-llama/llama-3.3-70b-instruct:free",
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
    )


def build_research_agent(
    llm: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
) -> AgentExecutor:
    model = llm or _load_llm()
    agent_tools = tools or [web_search_tool, calculator_tool, wikipedia_tool]

    agent = create_react_agent(model, agent_tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=agent_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
def run_query(query: str, agent: AgentExecutor) -> str:
    response = agent.invoke({"input": query})
    return response["output"]
