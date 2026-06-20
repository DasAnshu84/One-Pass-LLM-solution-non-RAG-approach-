# Force pure-Python implementation for protobuf to avoid Python 3.14 compatibility errors
import sys
sys.modules['google._upb._message'] = None
sys.modules['google.protobuf.pyext._message'] = None
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import re
import time
import textwrap
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Load environment variables
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_actual_api_key_here":
    print("[Warning] GEMINI_API_KEY is not configured or placeholder in .env file.")
else:
    genai.configure(api_key=api_key.strip())

MODEL = "gemini-2.5-flash"          # Use Gemini 2.5 Flash model
DOCS_DIR = Path("./docs")
TOP_K = 3                             # naive RAG chunks per query, kept small to mirror RAG cost

# -----------------------------------------------------------------------------
# SAMPLE CORPUS (auto-created on first run if PDF is not present)
# -----------------------------------------------------------------------------

SAMPLE_DOCS = {
    "Financial_Delegation_Schedule.txt": """\
FINANCIAL DELEGATION SCHEDULE - DEFENCE PROCUREMENT

Section 1: General Delegation of Powers
1.1 The Deputy Director (Procurement) may approve procurement proposals up to INR 50 lakh.
1.2 The Director (Procurement) may approve procurement proposals up to INR 2 crore.
1.3 The Additional Secretary may approve procurement proposals exceeding INR 2 crore,
    subject to Finance Committee concurrence.
1.4 All delegated powers under this schedule are subject to availability of budget and
    adherence to the Procurement Procedure Manual.
""",
    "Procurement_Procedure_Manual.txt": """\
PROCUREMENT PROCEDURE MANUAL

Section 4: Sole-Source Procurement
4.1 Sole-source procurement is permitted only where a single OEM (Original Equipment
    Manufacturer) is technically qualified to supply the item.
4.2 Sole-source procurement requests must be supported by a Technical Justification Note
    and a Cost Reasonableness Certificate.
4.3 Sole-source procurement proposals above INR 1 crore require additional vetting by the
    Internal Audit Wing before approval by the competent financial authority defined in the
    Financial Delegation Schedule.

Section 7: Naval Spares Procurement
7.1 Procurement of naval spares classified as "Critical Operational Spares" follows the
    sole-source procedure under Section 4 where OEM exclusivity applies.
""",
    "Threshold_Exceptions_Policy.txt": """\
THRESHOLD AND EXCEPTIONS POLICY

Section 2: Enhanced Threshold for Critical Operational Spares
2.1 For "Critical Operational Spares" as classified under the Procurement Procedure Manual,
    the monetary threshold for Director (Procurement) approval is enhanced from INR 2 crore
    to INR 3 crore, provided the Internal Audit Wing vetting under Section 4.3 of the
    Procurement Procedure Manual has been completed.
2.2 This enhanced threshold does not apply to routine spares procurement.
""",
}

# Questions matching the PDF corpus (Pages 10, 11, 12 of RegsNavyIII.pdf)
NAVY_QUESTIONS = [
    "What does the National Flag hoisted at the main indicate, and when is it allowed to be hoisted there?",
    "Under what conditions must an Indian Naval ship and all escorting ships wear the National Flag at the Jack staff and keep the white ensign and National Flag hoisted continuously day and night?",
    "How many guns are fired for a salute to the President in India, and what are the specific exceptions/circumstances when a 21-gun salute is fired instead of a 31-gun salute?"
]

# Fallback questions for the synthetic policy corpus
SYNTHETIC_QUESTIONS = [
    "Up to what amount can the Deputy Director (Procurement) approve procurement proposals?",
    "What two documents must support a sole-source procurement request?",
    "Who can approve a sole-source procurement of INR 2.5 crore for naval spares "
    "classified as Critical Operational Spares, assuming Internal Audit Wing vetting "
    "has been completed?",
]


def ensure_sample_corpus():
    DOCS_DIR.mkdir(exist_ok=True)
    
    # Check if we have the Navy Regulations PDF
    pdf_path = DOCS_DIR / "RegsNavyIII.pdf"
    if pdf_path.exists():
        # Clear any existing synthetic text files to avoid mixing
        for name in SAMPLE_DOCS.keys():
            txt_file = DOCS_DIR / name
            if txt_file.exists():
                txt_file.unlink()
                
        # Extract pages 10, 11, and 12 (indices 9, 10, 11 in pypdf)
        from pypdf import PdfReader
        print(f"[setup] Extracting pages 10, 11, and 12 from {pdf_path.name}...")
        reader = PdfReader(pdf_path)
        
        target_pages = [9, 10, 11]
        for idx in target_pages:
            page_num = idx + 1
            page_text = reader.pages[idx].extract_text()
            page_file = DOCS_DIR / f"RegsNavy_Page{page_num}.txt"
            page_file.write_text(page_text, encoding="utf-8")
            print(f"[setup] Saved page {page_num} to {page_file.name} ({len(page_text)} chars)")
    else:
        # Fallback to synthetic sample docs if PDF is missing
        print(f"[setup] {pdf_path.name} not found. Creating synthetic policy documents...")
        # Clear any existing RegsNavy files if we fall back
        for path in DOCS_DIR.glob("RegsNavy_Page*.txt"):
            path.unlink()
            
        for name, content in SAMPLE_DOCS.items():
            (DOCS_DIR / name).write_text(content, encoding="utf-8")


def load_docs():
    docs = {}
    for path in sorted(DOCS_DIR.glob("*.txt")):
        docs[path.name] = path.read_text(encoding="utf-8")
    return docs


# -----------------------------------------------------------------------------
# GEMINI API CALL HELPER WITH RETRY & RATE-LIMIT HANDLING
# -----------------------------------------------------------------------------

def call_gemini_with_retry(prompt, max_tokens=300, response_mime_type=None, retries=5, initial_backoff=5):
    """
    Calls the Gemini API with rate-limiting handling, automatic backoff,
    and a small polite sleep to avoid spamming the free tier API keys.
    """
    model = genai.GenerativeModel(MODEL)
    config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        response_mime_type=response_mime_type
    )
    
    backoff = initial_backoff
    for attempt in range(retries):
        try:
            resp = model.generate_content(prompt, generation_config=config)
            
            # Add diagnostics to understand truncation
            if resp.candidates:
                candidate = resp.candidates[0]
                finish_reason = candidate.finish_reason
                reason_name = getattr(finish_reason, "name", str(finish_reason))
                
                # Check safety ratings
                safety_blocked = False
                if hasattr(candidate, "safety_ratings"):
                    for rating in candidate.safety_ratings:
                        if rating.blocked:
                            safety_blocked = True
                
                # Check for truncated text
                text_len = len(resp.text) if resp.text else 0
                if reason_name != "STOP" or safety_blocked or text_len < 10:
                    print(f"[debug] Response status: finish_reason={reason_name}, safety_blocked={safety_blocked}, text_len={text_len}")
                    if safety_blocked and hasattr(candidate, "safety_ratings"):
                        print(f"[debug] Blocked safety ratings: {[r for r in candidate.safety_ratings if r.blocked]}")
            
            # Add a small delay between requests to remain well within free tier limits
            time.sleep(2)
            return resp.text
        except ResourceExhausted as e:
            if attempt == retries - 1:
                raise e
            print(f"[rate-limit] Quota limit hit. Retrying in {backoff} seconds (attempt {attempt + 1}/{retries})...")
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            if attempt == retries - 1:
                raise e
            print(f"[error] API request failed: {e}. Retrying in {backoff} seconds (attempt {attempt + 1}/{retries})...")
            time.sleep(backoff)
            backoff *= 2


# -----------------------------------------------------------------------------
# 1) NAIVE RAG BASELINE: chunk -> TF-IDF vectors -> cosine top-k -> answer
# -----------------------------------------------------------------------------

def chunk_docs(docs, min_len=40):
    """Split each doc into paragraph-ish chunks (blank-line separated)."""
    chunks = []  # list of (doc_name, chunk_text)
    for name, text in docs.items():
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for p in parts:
            if len(p) >= min_len:
                chunks.append((name, p))
    return chunks


def naive_rag_answer(question, chunks, top_k=TOP_K):
    texts = [c[1] for c in chunks]
    vectorizer = TfidfVectorizer().fit(texts + [question])
    chunk_vecs = vectorizer.transform(texts)
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, chunk_vecs)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    retrieved = [chunks[i] for i in top_idx]

    context = "\n\n".join(f"[{name}]\n{text}" for name, text in retrieved)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context is insufficient, say what's missing.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    
    answer = call_gemini_with_retry(prompt, max_tokens=300)
    retrieved_docs = sorted(set(name for name, _ in retrieved))
    return answer, retrieved_docs


# -----------------------------------------------------------------------------
# 2) STRUCTURED FACT-TABLE: per-doc extraction -> filtered lookup -> answer
# -----------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
Extract every distinct rule, condition, regulation, threshold, definition or authority statement from
the document below into a JSON list. Each item must have:
  - "rule": one-sentence plain statement of the rule or definition
  - "source_clause": the section/clause number it comes from (e.g. "1.1", "2(a)", "5", "6(1)")
Return ONLY a JSON list, no other text.

DOCUMENT ({doc_name}):
{doc_text}
"""


def extract_facts(docs):
    fact_table = []
    for name, text in docs.items():
        prompt = EXTRACTION_PROMPT.format(doc_name=name, doc_text=text)
        # Use application/json to enforce structured outputs from Gemini
        raw = call_gemini_with_retry(
            prompt, 
            max_tokens=800, 
            response_mime_type="application/json"
        )
        raw = raw.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            rules = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[warn] Could not parse extraction for {name}, skipping. Raw:\n{raw}")
            continue
        for r in rules:
            r["source_doc"] = name
            fact_table.append(r)
    return fact_table


def structured_answer(question, fact_table):
    # Small corpus -> pass the whole compact fact table (cheap: facts are short).
    # At larger scale, add a keyword/BM25 filter step here before this call.
    facts_text = "\n".join(
        f"- ({f['source_doc']} {f.get('source_clause','?')}): {f['rule']}"
        for f in fact_table
    )
    prompt = (
        "Answer the question using ONLY the facts below. Cite the source doc and "
        "clause for each fact you use. If facts conflict or one modifies another, "
        "resolve it explicitly and explain why.\n\n"
        f"FACTS:\n{facts_text}\n\nQUESTION: {question}"
    )
    
    return call_gemini_with_retry(prompt, max_tokens=400)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    ensure_sample_corpus()
    docs = load_docs()
    print(f"[setup] Loaded {len(docs)} docs: {', '.join(docs.keys())}\n")

    chunks = chunk_docs(docs)
    print(f"[setup] Naive RAG: {len(chunks)} chunks indexed (TF-IDF)\n")

    # Check which corpus was loaded to select corresponding questions
    is_navy = any("RegsNavy" in name for name in docs.keys())
    questions = NAVY_QUESTIONS if is_navy else SYNTHETIC_QUESTIONS

    print("[setup] Extracting structured fact table (one LLM call per doc)...")
    fact_table = extract_facts(docs)
    print(f"[setup] Extracted {len(fact_table)} rules total.\n")
    print("=" * 80)

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}\n")

        rag_answer, rag_docs = naive_rag_answer(q, chunks)
        print("--- [NAIVE RAG] ---")
        print(f"Retrieved from: {rag_docs}")
        print(textwrap.fill(rag_answer, 100))

        struct_answer = structured_answer(q, fact_table)
        print("\n--- [STRUCTURED FACT-TABLE] ---")
        print(textwrap.fill(struct_answer, 100))

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
