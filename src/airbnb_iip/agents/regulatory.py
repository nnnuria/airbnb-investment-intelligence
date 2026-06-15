"""
Regulatory Agent — RAG pipeline for STR regulatory questions.
src/airbnb_iip/agents/regulatory.py

Modern langchain implementation (no RetrievalQA dependency).

Flow:
    load PDFs → chunk → multilingual embeddings → FAISS index
    → retrieve relevant chunks → Gemini answers in English with citations
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env so GEMINI_API_KEY is available
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parents[3]
CORPUS_DIR = ROOT / "Data" / "regulatory"
INDEX_DIR  = ROOT / "data" / "processed" / "regulatory_index"

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100

HIGH_RISK_KEYWORDS = [
    "moratoria", "moratorium", "no new", "not possible", "prohibited",
    "suspended", "banned", "expire", "elimination", "sin licencia",
    "suspensión", "prohibido", "no se conceden", "no nuevas",
    "will not renew", "cannot obtain",
]
MEDIUM_RISK_KEYWORDS = [
    "restricted", "limited", "licence required", "licencia", "autorización",
    "declaración responsable", "requires approval", "saturated", "saturada",
    "prior licence",
]

SYSTEM_PROMPT = """You are a regulatory assistant for the Airbnb Investment
Intelligence Platform. Answer questions about short-term rental (STR)
regulations in Spanish cities, based ONLY on the documents provided.
Always answer in English.

Rules:
- Answer ONLY from the context. Do not invent rules.
- Always cite the source document name.
- If not in context say: "I could not find this in the official documents.
  Please verify directly with the relevant authority."
- End every answer with:
  "⚠️ Indicative only — not legal advice. Verify against current
   official municipal/regional sources."

Context:
{context}

Question: {question}

Answer:"""


class RegulatoryAgent:
    """
    RAG-based agent that answers STR regulatory questions with citations
    and returns a structured risk flag for the UC2 recommendation engine.
    """

    def __init__(self, rebuild_index: bool = False):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Add it to your .env file: GEMINI_API_KEY=your-key-here"
            )

        # Lazy imports to avoid top-level version conflicts
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.FAISS = FAISS
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = self._load_or_build_index(rebuild_index)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=500,
        )

    # ── Index ─────────────────────────────────────────────────────────────────

    def _load_or_build_index(self, rebuild: bool):
        from langchain_community.vectorstores import FAISS

        if INDEX_DIR.exists() and not rebuild:
            print(f"[Regulatory] Loading cached index from {INDEX_DIR}")
            return FAISS.load_local(
                str(INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._build_index()

    def _build_index(self):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS

        print(f"[Regulatory] Building index from {CORPUS_DIR} ...")

        pdf_files = list(CORPUS_DIR.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDFs found in {CORPUS_DIR}. "
                "Please populate Data/regulatory/ first."
            )
        print(f"[Regulatory] Found {len(pdf_files)} PDF(s)")

        all_docs = []
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                city = pdf_path.parent.name
                for doc in docs:
                    doc.metadata["city"] = city
                    doc.metadata["source_file"] = pdf_path.name
                all_docs.extend(docs)
                print(f"  ✓ {pdf_path.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"  ✗ Could not load {pdf_path.name}: {e}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(all_docs)
        print(f"[Regulatory] {len(chunks)} chunks created")

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
        print(f"[Regulatory] Index saved to {INDEX_DIR}")

        return vectorstore

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """
        Ask a regulatory question. Returns:
            answer      — English answer with citations
            sources     — list of source documents used
            risk_flag   — "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
            city        — city detected from sources
            disclaimer  — always present
        """
        # 1. Retrieve top-4 relevant chunks
        docs = self.vectorstore.similarity_search(question, k=4)

        # 2. Build context string with source labels
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source_file", "unknown")
            city   = doc.metadata.get("city", "unknown")
            context_parts.append(
                f"[Source: {city.title()} — {source}]\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 3. Build prompt and call Gemini
        prompt = SYSTEM_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # 4. Build citation list
        sources = list({
            f"{doc.metadata.get('city', 'unknown').title()} — "
            f"{doc.metadata.get('source_file', 'unknown')}"
            for doc in docs
        })

        # 5. Detect city
        cities = list({
            doc.metadata.get("city", "unknown")
            for doc in docs
            if doc.metadata.get("city") != "national"
        })
        city = cities[0] if len(cities) == 1 else "multiple"

        # 6. Infer risk level
        answer_lower = answer.lower()
        if any(kw in answer_lower for kw in HIGH_RISK_KEYWORDS):
            risk_flag = "HIGH"
        elif any(kw in answer_lower for kw in MEDIUM_RISK_KEYWORDS):
            risk_flag = "MEDIUM"
        elif any(w in answer_lower for w in ["allowed", "permitted", "possible"]):
            risk_flag = "LOW"
        else:
            risk_flag = "UNKNOWN"

        return {
            "answer": answer,
            "sources": sources,
            "risk_flag": risk_flag,
            "city": city,
            "disclaimer": (
                "Indicative only — not legal advice. "
                "Verify against current official municipal/regional sources."
            ),
        }

    def get_risk_flag(self, city: str, neighbourhood: str = "") -> dict[str, Any]:
        """
        Convenience method for the UC2 financial engine.
        Returns the risk flag for a given city/neighbourhood.
        """
        question = (
            f"Can I legally start a new short-term rental (Airbnb) "
            f"in {neighbourhood + ', ' if neighbourhood else ''}{city}? "
            f"Is a licence available and what are the restrictions?"
        )
        result = self.query(question)
        return {
            "risk_flag": result["risk_flag"],
            "reason": result["answer"],
            "sources": result["sources"],
            "city": city,
            "disclaimer": result["disclaimer"],
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building regulatory agent ...\n")
    agent = RegulatoryAgent()

    test_questions = [
        "Can I legally start a new Airbnb in central Madrid?",
        "What happens to Barcelona tourist flat licences in 2028?",
        "Is there a moratorium on new tourist licences in Málaga?",
        "Do I need a national registration number to list on Airbnb in Spain?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = agent.query(q)
        print(f"Risk: {result['risk_flag']}")
        print(f"A: {result['answer'][:400]}")
        print(f"Sources: {result['sources']}")
        print("-" * 60)