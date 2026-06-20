import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader

load_dotenv()

DOCUMENTS_DIR = "documents"
VECTORSTORE_DIR = "vectorstore"

# Embedding model - runs locally, free, no API key needed
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def build_vectorstore():
    """Loads all PDFs, chunks them, embeds them, and saves a FAISS index."""
    print("Loading documents...")
    all_docs = []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DOCUMENTS_DIR, filename)
            print(f"  Loading {filename}...")
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
            all_docs.extend(docs)
        elif filename.endswith(".txt"):
            path = os.path.join(DOCUMENTS_DIR, filename)
            print(f"  Loading {filename}...")
            loader = TextLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
            all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages total.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    # Build and save FAISS index
    print("Building vector store (this may take a minute)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to {VECTORSTORE_DIR}/")
    return vectorstore


def load_vectorstore():
    """Loads an existing FAISS index from disk."""
    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a document analysis agent for an insurance company.

Answer the user's question using ONLY the context provided below from internal documents.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain the answer, say "The documents do not contain information on this."
- Cite the source file when you reference specific information.
- Be concise and precise. Quote exact figures, thresholds, and policy language where relevant.

Context:
{context}"""),
    ("human", "{question}")
])

rag_chain = rag_prompt | llm


def query_rag(question: str, vectorstore=None, k: int = 4) -> str:
    """Retrieves relevant chunks and generates an answer."""
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Retrieve top k relevant chunks
    results = vectorstore.similarity_search(question, k=k)

    # Build context string with source tags
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source_file', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in results
    ])

    response = rag_chain.invoke({"context": context, "question": question})
    return response.content


if __name__ == "__main__":
    # First run: build the vector store
    if not os.path.exists(VECTORSTORE_DIR):
        build_vectorstore()
    else:
        print("Vector store already exists. Delete the 'vectorstore' folder to rebuild.")

    # Test queries
    test_questions = [
        "What are the unfair claims settlement practices an insurer must avoid?",
        "What does 'covered auto' mean under the business auto coverage form?",
        "What are the main business risks Chubb discloses?",
    ]

    print("\n" + "=" * 60)
    print("RAG AGENT TEST")
    print("=" * 60)

    vs = load_vectorstore()
    for q in test_questions:
        print(f"\nQ: {q}")
        print("-" * 60)
        answer = query_rag(q, vectorstore=vs)
        print(answer)