import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from db_connector import get_engine

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def get_sql_database():
    """Returns a LangChain SQLDatabase object pointed at the right engine."""
    engine, mode = get_engine()
    db = SQLDatabase(engine, include_tables=["insurance_claims"])
    return db, mode


def query_sql(question: str) -> str:
    """Takes a natural language question, generates SQL, runs it, returns answer."""
    db, mode = get_sql_database()
    SQL_AGENT_PREFIX = """You are a SQL agent working with an insurance claims database.

CRITICAL RULES:
1. The table 'insurance_claims' contains ALL claims (auto insurance only - there is no 
    separate "commercial" vs "personal" claim type column). Do not filter by a 
    "commercial auto" incident_type value - it does not exist in this data.
2. Before adding any WHERE filter, verify the exact column values exist by running a 
    quick SELECT DISTINCT on that column first if you're unsure.
3. Only use total_claim_amount for claim amount comparisons - never claim_amount or amount.
4. If a question mentions a dollar threshold from policy documents, apply it ONLY to 
    total_claim_amount and do not add unrelated filters unless explicitly asked.
5. Always double-check column names against the actual schema before filtering.
"""

    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        prefix=SQL_AGENT_PREFIX
    )

    result = agent.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    test_questions = [
        "How many claims were reported as fraud?",
        "What are the top 5 states by total claim amount?",
        "What is the average claim amount for major incidents vs minor incidents?",
        "How many claims involved police reports and were also flagged as fraud?",
        "Which auto make has the highest average total claim amount?"
    ]

    print("=" * 60)
    print("SQL AGENT TEST")
    print("=" * 60)

    for q in test_questions:
        print(f"\nQ: {q}")
        print("-" * 60)
        try:
            answer = query_sql(q)
            print(f"A: {answer}")
        except Exception as e:
            print(f"ERROR: {e}")