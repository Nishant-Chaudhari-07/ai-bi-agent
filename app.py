import streamlit as st
from graph import build_graph
import time

st.set_page_config(
    page_title="Enterprise AI Analyst Agent",
    page_icon="📊",
    layout="centered"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 14px;
        background: rgba(128,128,128,0.08);
        border-left: 3px solid rgba(128,128,128,0.3);
    }
    .pipeline-step.active {
        background: rgba(56,139,253,0.12);
        border-left: 3px solid #388bfd;
    }
    .pipeline-step.done {
        background: rgba(46,160,67,0.10);
        border-left: 3px solid #2ea043;
    }
    .metric-card {
        background: rgba(128,128,128,0.08);
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-number {
        font-size: 22px;
        font-weight: 700;
    }
    .metric-label {
        font-size: 12px;
        color: rgba(150,150,150,0.9);
        margin-top: 2px;
    }
    .source-badge {
        display: inline-block;
        background: rgba(56,139,253,0.15);
        color: #58a6ff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.title("📊 Enterprise AI Analyst Agent")
st.markdown(
    "**A multi-agent system that answers business questions by reading policy documents "
    "and querying live claims data — at the same time.**"
)
st.caption("Built for insurance & financial services analytics · LangChain · LangGraph · RAG · Snowflake")

st.divider()

# ---- Stats Row ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-number">4</div><div class="metric-label">Source documents</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-number">1,000</div><div class="metric-label">Claims records</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-number">4</div><div class="metric-label">AI agents</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-number">~30s</div><div class="metric-label">Avg. response time</div></div>', unsafe_allow_html=True)

st.write("")

# ---- About / How it works ----
with st.expander("ℹ️ How this system works", expanded=False):
    st.markdown("""
This system solves a real problem in insurance and financial services: business questions
often require BOTH a policy lookup AND a database query — normally done manually, by
different people, taking hours.

**The pipeline:**

1. **Router Agent** reads your question and decides what's needed — document search, a database query, or both
2. **RAG Agent** searches a vector index built from real insurance/regulatory documents
3. **SQL Agent** writes and executes SQL against a live database, self-correcting on errors
4. **Synthesis Agent** combines everything into one clear answer

**Source documents indexed:**
""")
    st.markdown("""
<span class="source-badge">Chubb Ltd 10-K (SEC filing)</span>
<span class="source-badge">NAIC Claims Settlement Act</span>
<span class="source-badge">ISO Commercial Auto Policy</span>
<span class="source-badge">Internal Claims Escalation Policy</span>
    """, unsafe_allow_html=True)

st.divider()

# ---- Session State Setup ----
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

# ---- Example Questions ----
st.markdown("**Try an example:**")
example_questions = [
    ("⚖️ + 📊", "What is the threshold for supervisor approval on commercial claims, and how many current claims exceed it?"),
    ("📊", "How many claims were reported as fraud?"),
    ("⚖️", "What does the NAIC say about unfair claims settlement practices?"),
    ("📊", "What are the top states by total claim amount?"),
]

cols = st.columns(2)
for i, (icon, q) in enumerate(example_questions):
    with cols[i % 2]:
        if st.button(f"{icon}  {q}", key=f"example_{i}", use_container_width=True):
            st.session_state.question_text = q

st.caption("⚖️ = needs document search &nbsp;&nbsp; 📊 = needs database query &nbsp;&nbsp; ⚖️+📊 = needs both")

st.divider()

# ---- Question Input ----
question = st.text_input(
    "Ask your own question:",
    key="question_text",
    placeholder="e.g. How many fraud claims are there in California?"
)

run_button = st.button("Ask the agents", type="primary", use_container_width=True)

# ---- Run the Agent Pipeline ----
if run_button and question:
    pipeline_placeholder = st.empty()

    steps = {
        "router": {"label": "Router Agent — classifying your question", "state": "pending"},
        "rag": {"label": "RAG Agent — searching policy documents", "state": "pending"},
        "sql": {"label": "SQL Agent — querying claims database", "state": "pending"},
        "synthesis": {"label": "Synthesis Agent — writing final answer", "state": "pending"},
    }

    def render_pipeline():
        html = ""
        for key, step in steps.items():
            cls = "pipeline-step"
            icon = "○"
            if step["state"] == "active":
                cls += " active"
                icon = "◐"
            elif step["state"] == "done":
                cls += " done"
                icon = "✓"
            html += f'<div class="{cls}">{icon} &nbsp; {step["label"]}</div>'
        pipeline_placeholder.markdown(html, unsafe_allow_html=True)

    render_pipeline()

    app = build_graph()
    final_state = None
    route_taken = None

    steps["router"]["state"] = "active"
    render_pipeline()

    for step in app.stream({"question": question}):
        node_name = list(step.keys())[0]
        node_output = step[node_name]

        if node_name == "router":
            steps["router"]["state"] = "done"
            route_taken = node_output["route"]
            if route_taken == "rag":
                steps["sql"]["state"] = "skipped_remove"
            elif route_taken == "sql":
                steps["rag"]["state"] = "skipped_remove"
            if route_taken in ("rag", "both"):
                steps["rag"]["state"] = "active"
            elif route_taken == "sql":
                steps["sql"]["state"] = "active"
            render_pipeline()

        elif node_name == "rag":
            steps["rag"]["state"] = "done"
            if route_taken == "both":
                steps["sql"]["state"] = "active"
            render_pipeline()

        elif node_name == "sql":
            steps["sql"]["state"] = "done"
            steps["synthesis"]["state"] = "active"
            render_pipeline()

        elif node_name == "synthesis":
            steps["synthesis"]["state"] = "done"
            final_state = node_output
            render_pipeline()

    # Clean up skipped steps from final display
    final_html = ""
    for key, step in steps.items():
        if step["state"] == "skipped_remove":
            continue
        cls = "pipeline-step done"
        final_html += f'<div class="{cls}">✓ &nbsp; {step["label"]}</div>'
    pipeline_placeholder.markdown(final_html, unsafe_allow_html=True)

    st.write("")
    st.subheader("📋 Answer")

    route_label = {
        "rag": "📄 Answered from policy documents",
        "sql": "🗄️ Answered from live claims data",
        "both": "⚖️ 📊 Answered using policy documents + live claims data"
    }.get(route_taken, "")
    st.caption(route_label)

    clean_answer = final_state["final_answer"].replace("`", "")
    st.markdown(
        f'<div style="background: rgba(128,128,128,0.06); border-radius: 10px; padding: 18px 20px; '
        f'border-left: 3px solid #2ea043; font-size: 15.5px; line-height: 1.6;">{clean_answer}</div>',
        unsafe_allow_html=True
    )

elif run_button and not question:
    st.warning("Please enter a question first.")

# ---- Footer ----
st.write("")
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Built by Nishant Chaudhari — Multi-Agent BI Assistant")
with col2:
    st.caption("[GitHub](https://github.com/Nishant-Chaudhari-07) · [Portfolio](https://nishantchaudhari.com)")