import os
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from router_agent import route_question
from rag_agent import query_rag, load_vectorstore
from sql_agent import query_sql

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ---- State Definition ----
class AgentState(TypedDict):
    question: str
    route: Optional[str]
    rag_answer: Optional[str]
    sql_answer: Optional[str]
    final_answer: Optional[str]


# ---- Nodes ----

def router_node(state: AgentState) -> AgentState:
    print(f"\n[ROUTER] Analyzing question...")
    route = route_question(state["question"])
    print(f"[ROUTER] → Route: {route.upper()}")
    return {**state, "route": route}


def rag_node(state: AgentState) -> AgentState:
    print(f"[RAG AGENT] Searching documents...")
    vs = load_vectorstore()
    answer = query_rag(state["question"], vectorstore=vs)
    print(f"[RAG AGENT] Done.")
    return {**state, "rag_answer": answer}


def sql_node(state: AgentState) -> AgentState:
    print(f"[SQL AGENT] Querying database...")

    # If RAG already ran and found relevant facts, pass them to SQL agent
    if state.get("rag_answer"):
        enriched_question = (
            f"{state['question']}\n\n"
            f"Context from policy documents (use any specific numbers/thresholds mentioned here "
            f"directly in your SQL query instead of searching for a threshold column in the database):\n"
            f"{state['rag_answer']}"
        )
    else:
        enriched_question = state["question"]

    answer = query_sql(enriched_question)
    print(f"[SQL AGENT] Done.")
    return {**state, "sql_answer": answer}


synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a synthesis agent for an enterprise insurance analytics system.

Combine the information below into a single, concise executive briefing that directly answers the user's question.

Rules:
- If both a policy answer and a data answer are present, connect them clearly (e.g. "Per policy X, the threshold is Y. Currently, Z records meet this condition.")
- Be concise - aim for 3-5 sentences.
- Do not repeat information unnecessarily.
- Maintain a professional, executive-briefing tone."""),
    ("human", """Question: {question}

Policy/Document Answer: {rag_answer}

Data/SQL Answer: {sql_answer}

Write the final synthesized answer.""")
])

synthesis_chain = synthesis_prompt | llm


def synthesis_node(state: AgentState) -> AgentState:
    print(f"[SYNTHESIS AGENT] Combining results...")

    rag_answer = state.get("rag_answer") or "Not retrieved."
    sql_answer = state.get("sql_answer") or "Not retrieved."

    # If only one source was used, skip synthesis LLM call - just pass it through
    if state["route"] == "rag":
        final = state["rag_answer"]
    elif state["route"] == "sql":
        final = state["sql_answer"]
    else:
        response = synthesis_chain.invoke({
            "question": state["question"],
            "rag_answer": rag_answer,
            "sql_answer": sql_answer
        })
        final = response.content

    print(f"[SYNTHESIS AGENT] Done.")
    return {**state, "final_answer": final}


# ---- Conditional Routing Logic ----

def route_decision(state: AgentState) -> str:
    """Decides which node(s) to go to after the router."""
    return state["route"]


# ---- Build the Graph ----

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("sql", sql_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("router")

    # After router, conditionally branch
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "sql": "sql",
            "both": "rag"  # for "both", go to rag first, then sql
        }
    )

    # If route was "both", after rag go to sql, then synthesis
    # If route was "rag" only, go straight to synthesis
    def after_rag(state: AgentState) -> str:
        return "sql" if state["route"] == "both" else "synthesis"

    graph.add_conditional_edges("rag", after_rag, {
        "sql": "sql",
        "synthesis": "synthesis"
    })

    graph.add_edge("sql", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


# ---- Main Entry Point ----

def ask(question: str) -> str:
    app = build_graph()
    result = app.invoke({"question": question})
    return result["final_answer"]


if __name__ == "__main__":
    test_questions = [
        "What is the threshold for supervisor approval on commercial claims, and how many current claims exceed it?",
        "How many claims were reported as fraud?",
        "What does the NAIC say about unfair claims settlement practices?",
    ]

    print("=" * 60)
    print("FULL AGENT GRAPH TEST")
    print("=" * 60)

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {q}")
        print('='*60)
        answer = ask(q)
        print(f"\nFINAL ANSWER:\n{answer}\n")