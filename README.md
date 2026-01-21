## RAG Chatbot (Retrieval-Augmented Generation)

A Retrieval-Augmented Generation (RAG) based chatbot that enables users to query their own documents/datasets using vector search + LLM generation.  
The system retrieves relevant context from a knowledge base and generates accurate answers grounded in the uploaded data.

---

## Key Features

- Document ingestion from custom dataset/documents
- Text chunking and embeddings generation
- Vector database indexing for similarity search
- Context retrieval during queries
- LLM-based answer generation using retrieved context
- Modular codebase (Data + Source separation)

---

## Project Structure

```bash
RAG_Chatbot/
│
├── Data/               # Knowledge base documents/dataset
├── Source/             # RAG pipeline and chatbot logic
├── .gitignore
└── README.md


## Tech Stack

Python
LangChain (RAG pipeline)
Vector Store: FAISS / Chroma / Pinecone (based on implementation)
Embeddings Model: OpenAI / HuggingFace
LLM: OpenAI GPT / Gemini / Llama (based on implementation)

## Workflow

•	Documents are loaded from the Data/ folder (or configured input source)
•	Text is split into chunks for better retrieval quality
•	Embeddings are generated for each chunk
•	Embeddings are stored in a vector database (indexing)
•	During a user query:
	Retrieve top relevant chunks from the vector database
	Combine retrieved context with the query prompt
	Generate final answer using the LLM

## Installation and Setup

1. Clone Repository

    git clone https://github.com/Hariganesh2505/RAG_Chatbot.git
    cd RAG_Chatbot

2. Create Virtual Environment

    python -m venv env

3. Activate Environment

    Windows:
        env\Scripts\activate
    Linux/Mac:
        source env/bin/activate

4. Install Dependencies

    pip install -r requirements.txt

## Environment Variables

Create a .env file in the root directory and add your API keys.

In env:

OPENAI_API_KEY=your_openai_api_key
# PINECONE_API_KEY=your_pinecone_api_key   (if applicable)
# GOOGLE_API_KEY=your_google_api_key       (if applicable)

## Running the Project

    Run the main entry script (based on your implementation):
    python Source/main.py

## If Streamlit UI exists:

    streamlit run Source/app.py

## Customization
You can modify:

    Chunk size and chunk overlap
    Embeddings model (OpenAI / HuggingFace)
    Vector database backend (FAISS / Chroma / Pinecone)
    LLM model selection

## Future Improvements

    Add UI support for document upload and management
    Add citations/source highlighting from retrieved chunks
    Support multiple document formats (PDF/DOCX/TXT)
    Cloud deployment support (AWS / Render / HuggingFace Spaces)


## Steps:

    Fork the repository
    Create a feature branch
    Commit your changes
    Push to the branch
    Create a Pull Request

## Author
Hari Ganesh S
GitHub: https://github.com/Hariganesh2505
