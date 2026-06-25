"""
Regulatory Agent — RAG pipeline for STR regulatory questions.
src/airbnb_iip/agents/regulatory.py

Modern langchain implementation (no RetrievalQA dependency).

Flow:
    load PDFs → chunk → multilingual embeddings → FAISS index
    → retrieve relevant chunks → Gemini answers in English with citations
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env so GEMINI_API_KEY is available
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parents[3]
CORPUS_DIR = ROOT / "Data" / "regulatory"
INDEX_DIR  = ROOT / "Data" / "processed" / "regulatory_index"

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100

# Chunks fed to the LLM per query. 8 (was 4) gives Barcelona/Málaga enough
# context to commit to a regime assessment instead of hedging or bailing out.
TOP_K = 8

# intfloat/multilingual-e5-* is trained to require these prefixes; without them
# retrieval quality collapses (verified: a Madrid query returned only cookie
# banners). The prefix lives in the embedding only, never in the stored text.
E5_QUERY_PREFIX   = "query: "
E5_PASSAGE_PREFIX = "passage: "

# Bumped when the embedding/indexing contract changes, to force a rebuild of a
# stale on-disk index (e.g. one built before the e5 prefixes were added).
INDEX_VERSION = "e5-prefixed-v2"

# Web-export PDFs (the national doc + "Business Channel"/"City Council" pages)
# leak boilerplate chunks — cookie banners, nav chrome, CSS selectors — that
# out-rank real legal text because they're short and generic. Dropped at index
# build time.
_BOILERPLATE_MARKERS = (
    "this website is managed by",
    "third-party cookies",
    "uses its own and/or",
    "attr(href)",
    "skip to main content",
    "accept cookies",
    "cookie policy",
)

# The corpus is Spanish; queries are issued in Spanish for same-language
# retrieval (e5 cross-lingual EN→ES over dense legalese underperforms badly).
# The LLM still answers in English from the Spanish context.
_TRANSLATE_PROMPT = (
    "Translate this short-term-rental regulatory question into Spanish for a "
    "document search. Output ONLY the Spanish translation, no preamble.\n\n"
    "Question: {q}\nSpanish:"
)

# Risk is classified by the LLM against an explicit rubric rather than by
# sniffing keywords out of the prose (which mislabels "...are not permitted, but
# in commercial buildings STRs are allowed" as LOW because it spots "allowed").
# The keyword heuristic (_keyword_risk) remains as a fail-soft fallback only.
_RISK_PROMPT = (
    "You classify short-term-rental (STR) regulatory RISK for a property "
    "investor, based ONLY on the regulatory answer below.\n\n"
    "Output EXACTLY ONE word — one of: HIGH, MEDIUM, LOW, UNKNOWN.\n\n"
    "Rubric:\n"
    "- HIGH: for a NEW entrant, new STR licences are unavailable, suspended, "
    "prohibited, capped/saturated, under a moratorium, or otherwise effectively "
    "impossible; or new STRs in residential/dispersed buildings are not "
    "permitted.\n"
    "- MEDIUM: STRs are allowed but materially restricted — licence/registration "
    "required, zoning or saturation limits, conditions, or area-dependent rules.\n"
    "- LOW: STRs are broadly allowed with only standard registration and no major "
    "barrier to a new operator.\n"
    "- UNKNOWN: the answer lacks enough information to decide.\n\n"
    "When the answer mixes permission and prohibition, classify by the hardest "
    "constraint a NEW entrant faces, not the most lenient.\n\n"
    "Answer:\n{answer}\n\n"
    "Question: {question}\n\nRISK:"
)

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
regulations in Spanish cities, based ONLY on the documents provided. The context
may be in Spanish or Catalan; always answer in English.

Rules:
- Answer ONLY from the context. Do not invent rules or numbers.
- When the context contains regulation relevant to the question, COMMIT to a
  definitive assessment — do not hedge or defer. State clearly whether a new STR
  can be licensed/registered, and any licence, registration, moratorium,
  saturation cap, or zoning/area restriction that applies to a new operator.
- Do NOT refuse just because the context is partial — summarise what it does
  say and cite it.
- Only when the context contains NOTHING relevant to the question, say exactly:
  "I could not find this in the official documents.
  Please verify directly with the relevant authority."
- Always cite the source document name(s).
- End every answer with:
  "⚠️ Indicative only — not legal advice. Verify against current
   official municipal/regional sources."

Context:
{context}

Question: {question}

Answer:"""


def _make_e5_embeddings():
    """Build the e5 embeddings, wrapped to add 'query:'/'passage:' prefixes.

    The prefix is applied only inside ``embed_documents`` / ``embed_query`` so
    the FAISS-stored ``page_content`` stays clean (no prefix leaks into the
    context shown to the LLM). Imports are local to keep module import cheap.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.embeddings import Embeddings

    base = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    class _E5PrefixEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return base.embed_documents([E5_PASSAGE_PREFIX + t for t in texts])

        def embed_query(self, text: str) -> list[float]:
            return base.embed_query(E5_QUERY_PREFIX + text)

    return _E5PrefixEmbeddings()


def _is_boilerplate(text: str) -> bool:
    """True if a chunk is web-export chrome rather than legal/regulatory prose."""
    low = text.lower()
    if any(m in low for m in _BOILERPLATE_MARKERS):
        return True
    # Mostly non-prose (CSS selectors, URL/date dumps): low letter+space ratio.
    if len(text) >= 40:
        prose = sum(c.isalpha() or c.isspace() for c in text)
        if prose / len(text) < 0.6:
            return True
    return False


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
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.FAISS = FAISS
        self.embeddings = _make_e5_embeddings()
        self._tx_cache: dict[str, str] = {}   # query → Spanish translation
        self.vectorstore = self._load_or_build_index(rebuild_index)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=500,
            # Without this, gemini-2.5-flash spends max_output_tokens on
            # internal reasoning and the visible answer gets cut off
            # mid-sentence (finish_reason="MAX_TOKENS") even for short
            # prompts. Confirmed via a raw API smoke test.
            thinking_budget=0,
        )

    # ── Index ─────────────────────────────────────────────────────────────────

    def _load_or_build_index(self, rebuild: bool):
        from langchain_community.vectorstores import FAISS

        meta_path = INDEX_DIR / "build_meta.json"
        fresh = False
        if meta_path.exists():
            try:
                fresh = json.loads(meta_path.read_text()).get("version") == INDEX_VERSION
            except (json.JSONDecodeError, OSError):
                fresh = False

        if INDEX_DIR.exists() and fresh and not rebuild:
            print(f"[Regulatory] Loading cached index from {INDEX_DIR}")
            return FAISS.load_local(
                str(INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        if INDEX_DIR.exists() and not fresh:
            print(f"[Regulatory] Index missing/stale (need {INDEX_VERSION!r}) — rebuilding")
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
                print(f"  [OK] {pdf_path.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"  [FAIL] Could not load {pdf_path.name}: {e}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(all_docs)
        n_raw = len(chunks)
        chunks = [c for c in chunks if not _is_boilerplate(c.page_content)]
        print(f"[Regulatory] {len(chunks)} chunks kept "
              f"({n_raw - len(chunks)} boilerplate chunks dropped)")

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
        (INDEX_DIR / "build_meta.json").write_text(json.dumps({
            "version": INDEX_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "chunks": len(chunks),
        }, indent=2))
        print(f"[Regulatory] Index saved to {INDEX_DIR}")

        return vectorstore

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self, question: str, city: str | None = None, translate: bool = True,
    ) -> dict[str, Any]:
        """
        Ask a regulatory question. Returns:
            answer      — English answer with citations
            sources     — list of source documents used
            risk_flag   — "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
            city        — city detected from sources
            disclaimer  — always present

        ``city`` restricts retrieval to that city's documents + national
        rules. Without it, a city-specific FAQ written in a more
        question-like style (e.g. Málaga's) can outrank the actually
        relevant city's denser legal/decree text for a generic-sounding
        question — a structural weakness of dense retrieval over a
        heterogeneous, mixed-genre corpus, not a missing-document problem.
        Pass ``city`` whenever it's already known (see :meth:`get_risk_flag`).
        """
        # 1. Retrieve the top-TOP_K relevant chunks, optionally city-scoped.
        # Filtered manually (search everything, then keep matches) rather
        # than via FAISS's filter= kwarg: the corpus is small (low hundreds
        # of chunks) so an exhaustive search is cheap, and this sidesteps
        # langchain_community's filter/fetch_k interaction being easy to
        # mis-tune (verified during development that it under-returns
        # unless fetch_k is raised to cover the full corpus anyway).
        # The corpus is Spanish — retrieve in Spanish (the LLM still answers in
        # English from the retrieved context). Skip via translate=False if the
        # caller already passes a Spanish query.
        retrieval_q = self._translate_for_retrieval(question) if translate else question

        if city:
            target_city = city.strip().lower()
            total_chunks = self.vectorstore.index.ntotal
            ranked = self.vectorstore.similarity_search(retrieval_q, k=total_chunks)
            docs = [
                d for d in ranked if d.metadata.get("city") in (target_city, "national")
            ][:TOP_K]
        else:
            docs = self.vectorstore.similarity_search(retrieval_q, k=TOP_K)

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

        # 6. Classify risk — LLM rubric, with the keyword heuristic as fallback.
        risk_flag = self._classify_risk(question, answer)

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

    def _translate_for_retrieval(self, text: str) -> str:
        """Translate a question to Spanish for retrieval (cached, fail-soft).

        Identical queries (e.g. the templated risk question per city) are cached
        so we don't pay a Gemini call per repeat. On any failure we fall back to
        the original text — degraded retrieval beats a crash.
        """
        if text in self._tx_cache:
            return self._tx_cache[text]
        try:
            resp = self.llm.invoke(_TRANSLATE_PROMPT.format(q=text))
            out = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            result = out or text
        except Exception as exc:
            logger.warning("[Regulatory] translation failed (%s); using original", exc)
            result = text
        self._tx_cache[text] = result
        return result

    _RISK_LEVELS = ("HIGH", "MEDIUM", "LOW", "UNKNOWN")

    def _classify_risk(self, question: str, answer: str) -> str:
        """Classify HIGH/MEDIUM/LOW/UNKNOWN via the LLM rubric (fail-soft).

        Falls back to the keyword heuristic if the LLM call fails or returns an
        unrecognised label — a degraded label beats crashing the decision engine.
        """
        try:
            resp = self.llm.invoke(_RISK_PROMPT.format(answer=answer, question=question))
            out = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()
            # Exact leading token first, then any mention in rubric priority order.
            head = out.split()[0] if out.split() else ""
            if head in self._RISK_LEVELS:
                return head
            for level in self._RISK_LEVELS:
                if level in out:
                    return level
        except Exception as exc:
            logger.warning("[Regulatory] risk classification failed (%s); "
                           "using keyword fallback", exc)
        return self._keyword_risk(answer)

    @staticmethod
    def _keyword_risk(answer: str) -> str:
        """Legacy keyword heuristic — fallback when the LLM classifier is unusable."""
        a = answer.lower()
        if any(kw in a for kw in HIGH_RISK_KEYWORDS):
            return "HIGH"
        if any(kw in a for kw in MEDIUM_RISK_KEYWORDS):
            return "MEDIUM"
        if any(w in a for w in ["allowed", "permitted", "possible"]):
            return "LOW"
        return "UNKNOWN"

    def get_risk_flag(self, city: str, neighbourhood: str = "") -> dict[str, Any]:
        """
        Convenience method for the UC2 financial engine.
        Returns the risk flag for a given city/neighbourhood.
        """
        # Regime-neutral phrasing: ask about licence/registration availability
        # and the specific barriers (moratorium / cap / zoning), not just
        # "restrictions" — the latter biased retrieval toward restriction-heavy
        # cities and starved permissive ones (Málaga) of relevant chunks.
        where = f"{neighbourhood + ', ' if neighbourhood else ''}{city}"
        question = (
            f"For a new short-term / tourist rental (vivienda de uso turístico) "
            f"in {where}: is a new licence or registration currently available "
            f"to a new operator? Are there moratoria, saturation caps, zoning "
            f"limits, or other restrictions on new STRs?"
        )
        result = self.query(question, city=city)
        return {
            "risk_flag": result["risk_flag"],
            "reason": result["answer"],
            "sources": result["sources"],
            "city": city,
            "disclaimer": result["disclaimer"],
        }


@lru_cache(maxsize=1)
def get_regulatory_agent() -> RegulatoryAgent:
    """Process-wide singleton — loads the embeddings model + FAISS index once.

    Without this, a caller that re-instantiates RegulatoryAgent on every
    call (e.g. a Streamlit page, which reruns its whole script on every
    interaction) reloads the embeddings model and FAISS index from scratch
    each time — slow, and needlessly re-reads from disk.
    """
    return RegulatoryAgent()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # LLM answers can contain emoji/accented characters; Windows' default
    # console encoding (cp1252) can't print them and crashes mid-loop.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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