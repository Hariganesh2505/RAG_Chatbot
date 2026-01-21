from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# -------------------------
# CONFIG
# -------------------------
INDEX_NAME = "enter your index name"   # change to your Pinecone index name
PDF_PATH = "enter your pdf path"              # optional: r"D:\RAG Model langchain\Source\your.pdf"

# Best: set Pinecone key in env (CMD/PowerShell)
# os.environ["PINECONE_API_KEY"] = "YOUR_KEY"

agent = None

def init_rag_agent():
    """
    Initializes your RAGRetrieval and Agent only once
    so Flask doesn't reload it every request.
    """
    global agent
    if agent is None:
        from Code.rag import RAGRetrieval, AIAgent  # must exist in same folder
        rag_retriever = RAGRetrieval(
            index_name=INDEX_NAME,
            pdf_path=PDF_PATH,
            top_k=6,
            verbose=True
        )
        agent = AIAgent(rag_retriever)


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")  # Chat UI


@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "RAG Flask API Running!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        init_rag_agent()

        data = request.get_json(force=True)
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is missing"}), 400

        answer = agent.run(query)

        return jsonify({
            "query": query,
            "answer": answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
