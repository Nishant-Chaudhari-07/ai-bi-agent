import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing agent inside an enterprise insurance analytics system.

Your job is to read a user's business question and decide which tool(s) are needed to answer it.

You have access to two tools:
1. RAG - searches through insurance policy documents, underwriting guidelines, and regulatory filings to find policy language, rules, definitions, and thresholds.
2. SQL - queries a Snowflake database containing insurance claims records with fields like claim_amount, incident_type, insured_type, incident_state, policy_number, fraud_flag, and severity.

Routing rules:
- If the question asks about policy language, definitions, thresholds, rules, or compliance → route to "rag"
- If the question asks about numbers, counts, trends, amounts, or patterns in claims data → route to "sql"
- If the question asks about a threshold or rule AND wants to see data related to it → route to "both"

Respond with ONLY one word: rag, sql, or both. Nothing else."""),
    ("human", "{question}")
])

router_chain = router_prompt | llm


def route_question(question: str) -> str:
    """Routes a business question to the appropriate agent(s)."""
    response = router_chain.invoke({"question": question})
    route = response.content.strip().lower()

    # Validate output
    if route not in ("rag", "sql", "both"):
        print(f"WARNING: Unexpected route '{route}', defaulting to 'both'")
        route = "both"

    return route


if __name__ == "__main__":
    # Test with 5 questions covering all three routes
    test_questions = [
        "What is the company's policy on claim escalation for high-severity incidents?",
        "How many claims were flagged as fraud in California last quarter?",
        "What is the threshold for supervisor approval on commercial claims, and how many current claims exceed it?",
        "Summarize the underwriting guidelines for fleet vehicle coverage.",
        "Show me the top 5 states by total claim amount for corporate policyholders."
    ]

    print("=" * 60)
    print("ROUTER AGENT TEST")
    print("=" * 60)

    for q in test_questions:
        route = route_question(q)
        print(f"\nQ: {q}")
        print(f"→ Route: {route.upper()}")