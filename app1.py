import streamlit as st
from graph import build_graph

st.set_page_config(
    page_title="Enterprise AI Analyst Agent",
    page_icon="📊",
    layout="centered"
)

# ---- Header ----
st.title("📊 Enterprise AI Analyst Agent")
st.caption("Multi-agent BI assistant for insurance operations — combines policy document search (RAG) with live data queries (SQL) through LangGraph orchestration.")

with st.expander("ℹ️ About this system"):
    st.markdown("""
    This system answers business questions by routing them to one or more specialized AI agents:

    - **RAG Agent** — searches internal policy documents, NAIC regulatory guidelines, and claims escalation policy
    - **SQL Agent** — queries a live insurance claims database (Snowflake, with SQLite fallback for demos)
    - **Router Agent** — decides which agent(s) are needed for each question
    - **Synthesis Agent** — combines results into one executive briefing

    Built with LangChain, LangGraph, FAISS, Groq (Llama 3.3 70B), and Snowflake.
    """)

st.divider()

# ---- Session State Setup ----
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

# ---- Example Questions ----
st.markdown("**Try an example:**")
example_questions = [
    "What is the threshold for supervisor approval on commercial claims, and how many current claims exceed it?",
    "How many claims were reported as fraud?",
    "What does the NAIC say about unfair claims settlement practices?",
    "What are the top states by total claim amount?",
]

cols = st.columns(2)
for i, q in enumerate(example_questions):
    with cols[i % 2]:
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.question_text = q

st.divider()

# ---- Question Input ----
question = st.text_input(
    "Ask a business question:",
    key="question_text",
    placeholder="e.g. How many fraud claims are there in California?"
)

run_button = st.button("Ask", type="primary", use_container_width=True)

# ---- Run the Agent Pipeline ----
if run_button and question:
    status_container = st.status("Processing your question...", expanded=True)

    with status_container:
        st.write("🧭 **Router** — analyzing question...")
        app = build_graph()

        # Stream through the graph to show progress
        final_state = None
        for step in app.stream({"question": question}):
            node_name = list(step.keys())[0]
            node_output = step[node_name]

            if node_name == "router":
                st.write(f"→ Route: **{node_output['route'].upper()}**")
            elif node_name == "rag":
                st.write("📄 **RAG Agent** — searched policy documents")
            elif node_name == "sql":
                st.write("🗄️ **SQL Agent** — queried claims database")
            elif node_name == "synthesis":
                st.write("✍️ **Synthesis Agent** — combining results...")
                final_state = node_output

        status_container.update(label="Complete", state="complete", expanded=False)

    st.divider()
    st.subheader("Answer")
    clean_answer = final_state["final_answer"].replace("`", "")
    st.markdown(clean_answer)

elif run_button and not question:
    st.warning("Please enter a question.")

# ---- Footer ----
st.divider()
st.caption("Built by Nishant Chaudhari — Multi-Agent BI Assistant, Project 2 of 2 in agentic AI portfolio series.")