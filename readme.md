# Enterprise AI Analyst Agent

![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LangGraph-green)

### 🔗 [Try the live demo →](https://ai-bi-agent-enterprise-analyst.streamlit.app/)

*Runs entirely on an embedded SQLite database — no login or setup required. Ask a business question and watch the multi-agent pipeline route it through document search, live data querying, or both.*

A multi-agent business intelligence assistant that answers natural language questions by orchestrating document retrieval (RAG) and live database queries (SQL) through LangGraph - built for the insurance and financial services domain.

## The Problem

Analysts at insurance and financial services firms routinely answer questions that span two disconnected systems: internal policy documents (PDFs, regulatory guidelines) and transactional data warehouses. A single business question like *"What's our claims escalation threshold, and how many current claims exceed it?"* requires manually searching documents AND writing SQL - often taking hours and involving multiple people.

This system answers that same question in under a minute, with the policy reference and live data in a single response.

## Architecture

User Question

     |

     v

Router Agent (LLM-based intent classification)

     |

  +--+--+

  |  |  |

  v  v  v

RAG SQL Both

  |  |  |

  +--+--+

     |

     v

Synthesis Agent

     |

     v

Final Answer

**Router Agent** - classifies each question as needing document search, database query, or both

**RAG Agent** - retrieves relevant context from a FAISS vector store built from real insurance policy documents (NAIC regulatory guidelines, ISO commercial auto policy forms, Chubb's SEC 10-K filing, and an internal claims escalation policy)

**SQL Agent** - writes and executes SQL against a live Snowflake warehouse, with a SQLite fallback for zero-setup demos. Self-corrects on schema errors without human intervention.

**Synthesis Agent** - combines outputs from RAG and SQL into a single executive briefing, passing extracted facts (like dollar thresholds) from documents directly into the SQL agent's context to prevent re-deriving numbers that already came from policy text

## Why Dual-Mode Database

Snowflake requires authentication, which isn't practical for anyone trying this project without credentials. The system auto-detects available credentials and falls back to a pre-loaded SQLite database so anyone can run it instantly with zero setup. Live Snowflake access uses key-pair authentication (no password, no MFA interruption) for fully automated operation.

## Tech Stack

- **Orchestration:** LangChain, LangGraph
- **LLM:** Groq (Llama 3.3 70B)
- **Vector Store:** FAISS with local sentence-transformer embeddings
- **Data Warehouse:** Snowflake (key-pair auth) with SQLite demo fallback
- **UI:** Streamlit
- **Documents:** Real regulatory and corporate filings (NAIC, ISO, SEC EDGAR)

## Dataset

- **Documents:** 4 real insurance/regulatory documents (~470+ pages combined) - Chubb Ltd's SEC 10-K filing, NAIC Unfair Claims Settlement Practices Act, ISO Business Auto Coverage Form, internal claims escalation policy
- **Claims data:** ~1,000 auto insurance claims records (Kaggle, 39 fields including claim amounts, fraud flags, incident details, geography)

## Key Engineering Decisions

- **Self-correcting SQL generation** - the SQL agent inspects schema errors and rewrites queries autonomously rather than failing
- **No hallucination policy** - when data genuinely isn't available, the system reports that honestly instead of fabricating an answer
- **Cross-agent context passing** - facts extracted by the RAG agent (e.g. dollar thresholds from policy text) are passed into the SQL agent's query generation step, so numeric values found in unstructured documents drive structured queries correctly
- **Local embeddings** - no external API calls for document embedding, relevant for data-sensitive regulated industries

## Running Locally

```bash
git clone [git clone https://github.com/Nishant-Chaudhari-07/ai-bi-agent.git]
cd ai-bi-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

No credentials needed - the app runs in SQLite demo mode automatically if Snowflake credentials aren't configured.

## Project Context

This is the second of two agentic AI projects built to demonstrate production-pattern multi-agent system design for business intelligence use cases. The first project, an AI-powered churn and revenue risk analyzer, focused on scheduled automation and LLM summarization. This project extends that into interactive, multi-step agent orchestration with retrieval-augmented generation and live data querying.

Built by Nishant Chaudhari

