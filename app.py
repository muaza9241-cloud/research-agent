"""Run the ReAct research agent on a set of test queries."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from agents.research_agent import build_research_agent, run_query

TEST_QUERIES = [
    "What are the latest news about AI space missions?",
    "Calculate 25 * 4 + 10.",
]

def main() -> None:
    print("Initializing ReAct research agent (search + calculator)...\n")
    agent = build_research_agent()

    for query in TEST_QUERIES:
        print(f"\n==================================================")
        print(f"Running Query: {query}")
        print(f"==================================================")
        output = run_query(query, agent)
        print(output)

if __name__ == "__main__":
    main()
