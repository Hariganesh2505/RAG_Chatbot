from Code.rag import RAG, RAGRetrieval, AIAgent
import os

# ---------------- PDF Upload to Pinecone ----------------
pdf_path = r"enter your pdf path"  # e.g., r"D:\RAG Model langchain\Source\your.pdf"
index_name = "enter your index name"
embedding_model = "enter your embedding model"  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
embed_dimension = "enter your embedding dimension"  # matches MiniLM L6 v2

# Initialize RAG uploader
rag = RAG(
    index_name=index_name,
    embedding_model=embedding_model,
    embed_dimension=embed_dimension,
    pdf_path=pdf_path,
    verbose=True
)

print("\nUploading PDF to Pinecone...")
upload_res = rag.load_into_pinecone()
print(upload_res)

# ---------------- RAG Retrieval ----------------
retriever = RAGRetrieval(
    index_name=index_name,
    embedding_model=embedding_model,
    llm_model="google/flan-t5-large",  # you can switch to faster/smaller model if needed
    top_k=3,
    pdf_path=pdf_path,
    verbose=True
)

# ---------------- AI Agent ----------------
agent = AIAgent(retriever)

# ---------------- Interactive Chat Loop ----------------
print("\nRAG Agent is ready. Type 'exit' to quit.")
while True:
    query = input(">> Enter your question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Run query through the agent
    try:
        answer = agent.run(query)
    except Exception as e:
        answer = f"Error during retrieval: {e}"

    print("\nAgent:", answer, "\n")
