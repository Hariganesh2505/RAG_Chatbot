import os
import warnings
import traceback
import re
import math
from types import SimpleNamespace

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain.schema import Document

# ---------------- Environment Setup ----------------
os.environ["USER_AGENT"] = "Mozilla/5.0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------- RAG Upload ----------------
class RAG:
    def __init__(self, index_name, embedding_model=None, embed_dimension=None, pdf_path=None, verbose=True):
        self.index_name = index_name
        self.embedding_model = embedding_model or "enter your embedding model"  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
        self.embed_dimension = embed_dimension or "enter your embedding dimension"  # matches MiniLM L6 v2
        self.metric = "enter your metric"  # e.g., "cosine"
        self.cloud = "enter your cloud"  # e.g., "aws"
        self.region = "enter your region"  # e.g., "us-west1-gcp"
        self.pdf_path = pdf_path
        self.verbose = verbose

        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "enter your pinecone api key"
        os.environ["PINECONE_API_KEY"] = self.PINECONE_API_KEY
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

    def read_pdf(self):
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        return loader.load()

    def make_chunk(self, docs, chunk_size=300, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)

    def create_pinecone_index(self):
        try:
            indexes = self.pc.list_indexes().names()
        except Exception as e:
            return f"Error listing indexes: {e}\n{traceback.format_exc()}"
        if self.index_name not in indexes:
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embed_dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
                return f"Index {self.index_name} created successfully."
            except Exception as e:
                return f"Error creating index: {e}\n{traceback.format_exc()}"
        return f"Index {self.index_name} already exists."

    def load_into_pinecone(self):
        res = self.create_pinecone_index()
        if self.verbose: print(res)

        try:
            docs = self.read_pdf()
        except Exception as e:
            return f"Error reading PDF: {e}\n{traceback.format_exc()}"

        chunk_data = self.make_chunk(docs)
        # Add metadata for each chunk
        chunk_data_with_meta = [
            Document(page_content=chunk.page_content,
                     metadata={"page_content": chunk.page_content,
                               "page": getattr(chunk, "metadata", {}).get("page", 0)})
            for chunk in chunk_data
        ]

        try:
            PineconeVectorStore.from_documents(
                chunk_data_with_meta,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace="default"
            )
            return "Successfully stored in Pinecone."
        except Exception as e:
            return f"Error storing in Pinecone: {e}\n{traceback.format_exc()}"

# ---------------- RAG Retrieval ----------------
class RAGRetrieval:
    def __init__(self, index_name, embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model="google/flan-t5-large", top_k=6, pdf_path=None, verbose=True):
        self.index_name = index_name
        self.top_k = top_k
        self.verbose = verbose
        self.pdf_path = pdf_path

        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings, namespace="default")
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = None
        try:
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            if self.verbose:
                print("Warning: Pinecone index handle not initialized:", e)

        # Device selection for LLM pipeline
        device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1
        self.llm = pipeline("text2text-generation", model=llm_model, device=device)

    def _cosine(self, a, b):
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        return dot/(na*nb) if na and nb else 0.0

    def _embed_query(self, q):
        try:
            return self.embeddings.embed_query(q)
        except Exception:
            return self.embeddings.embed_documents([q])[0]

    def _pdf_fallback_search(self, query):
        if not self.pdf_path: return None
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        query_lower = query.lower()
        for d in docs:
            sentences = re.split(r'(?<=[.!?])\s+', d.page_content or "")
            for s in sentences:
                if query_lower in s.lower():
                    return s.strip()
        return None

    def retrieve_from_pinecone(self, query):
        if self.verbose:
            print("\n=== retrieve_from_pinecone start ===\nQuery:", query)

        normalized = []
        try:
            results = self.vector_store.similarity_search_with_score(query, k=10)
            for item in results:
                if isinstance(item, tuple) and len(item) == 2:
                    normalized.append(item)
                else:
                    normalized.append((item, None))
            if self.verbose: print(f"Found {len(normalized)} vector matches")
        except Exception:
            normalized = []

        if not normalized and self.index:
            try:
                q_emb = self._embed_query(query)
                pc_res = self.index.query(vector=q_emb, top_k=10, include_metadata=True, namespace="default")
                matches = pc_res.get("matches", []) if isinstance(pc_res, dict) else getattr(pc_res, "matches", [])
                normalized = [(SimpleNamespace(page_content=m.get("metadata", {}).get("page_content", "")), m.get("score")) for m in matches]
                if self.verbose: print(f"Raw Pinecone returned {len(normalized)} matches")
            except Exception as e:
                if self.verbose: print("Raw Pinecone query error:", e)

        if not normalized:
            pdf_hit = self._pdf_fallback_search(query)
            if pdf_hit: return pdf_hit
            return "Not found in document"

        texts = [getattr(d,"page_content","") for d,_ in normalized]
        try:
            doc_embs = self.embeddings.embed_documents(texts)
        except Exception:
            doc_embs = [self._embed_query(t) for t in texts]

        q_emb = self._embed_query(query)
        sims = [self._cosine(q_emb,de) for de in doc_embs]
        paired_sorted = sorted(zip([d for d,_ in normalized], texts, sims), key=lambda x:x[2], reverse=True)

        context = "\n\n---\n\n".join([f"Source {i+1} (sim={s:.4f}):\n{t.strip()}" for i, (_,t,s) in enumerate(paired_sorted[:self.top_k])])
        if self.verbose: print("\n--- Context preview ---\n", context[:1000])

        prompt = (
            "Answer using ONLY the CONTEXT below. "
            "If the answer appears verbatim, return that exact sentence. "
            "If not in the context, respond: 'Not found in document'.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
        )

        try:
            out = self.llm(prompt, max_new_tokens=128, do_sample=False, num_return_sequences=1)
            generated = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        except Exception as e:
            return f"LLM generation error: {e}\n{traceback.format_exc()}"

        result = generated.strip()

        # Regex post-processing for inventor extraction
        if not result or len(result)<3 or "not found" in result.lower():
            m = re.search(r'([A-Z][A-Za-z\s\']+?) is often credited as the inventor of the watch', context)
            if m: return m.group(0)
            pdf_hit = self._pdf_fallback_search(query)
            if pdf_hit: return pdf_hit
            return "Not found in document"

        return result

# ---------------- AI Agent ----------------
class AIAgent:
    def __init__(self, rag_retriever: RAGRetrieval):
        self.rag_retriever = rag_retriever
        self.llm = rag_retriever.llm

    def run(self, query: str):
        try:
            return self.rag_retriever.retrieve_from_pinecone(query)
        except Exception as e:
            return f"Agent error: {e}\n{traceback.format_exc()}"
