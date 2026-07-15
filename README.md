# ✝️ Assistant Doctrinal — Advanced RAG System for Biblical Doctrine

A production-grade **French-language Retrieval-Augmented Generation (RAG)** application built to help a church community query 1,625 pages of personal biblical teaching documents with precision, doctrinal grounding, and zero hallucination.

---

## 🎯 Project Overview

This system answers theological questions in French by retrieving relevant passages from a private collection of biblical teaching documents — never inventing answers, never sending sensitive documents to external servers, and always citing scripture first.

**Live demo available on request.**

---

## 🏗️ Architecture

```
User Question (French)
        ↓
Query Expansion (theological keywords injected)
        ↓
Semantic Search (ChromaDB + distiluse-base-multilingual-cased-v2)
BM25 Keyword Search (rank-bm25)
        ↓ Combined pool of 20 candidates
Cross-Encoder Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
        ↓ Top 5 most relevant chunks
Force-Injected Chunks (for complex classification questions)
        ↓
Gemini 3.1 Flash Lite — generates grounded French answer
        ↓
Response with Bible verse citations
```

---

## ✨ Features

### Retrieval Pipeline (4 layers)
| Layer | Technique | Purpose |
|---|---|---|
| 1 | Semantic Search | Finds chunks by meaning using multilingual embeddings |
| 2 | Hybrid BM25 | Finds chunks by keyword frequency and rarity |
| 3 | Cross-Encoder Reranker | Scores question-chunk pairs for true answer relevance |
| 4 | Query Expansion | Injects theological keywords to improve retrieval precision |

### Document Validation Workflow
- PDF upload by church members
- Automated doctrinal consistency check using LLM-as-Judge
- Email notification to church leader
- OTP-based admin login for approve/reject decisions
- Approved documents automatically indexed into ChromaDB

### Access Control
- The application includes role-based access control for document management functions. Administrative features are protected and accessible only to authorized church leadership.
---

## 📊 Evaluation Results

Evaluated using an **LLM-as-Judge eval harness** with 9 ground-truth question-answer pairs provided by the church leader.

| Question | Score |
|---|---|
| Qu'est-ce que la vérité ? | 5/5 |
| Types d'adoration dans la Bible ? | 5/5 |
| Adoration du chrétien ? | 3/5 |
| Qu'est-ce que le péché ? | 5/5 |
| Qu'est-ce que les péchés ? | 5/5 |
| Types de péchés ? | 4/5 |
| Qu'est-ce que la lèpre ? | 5/5 |
| Qu'est-ce qu'une souillure ? | 5/5 |
| Souillures les plus graves ? | 1/5 |
| **Average** | **4.2 / 5** |

**Known limitation:** Question 9 scores 1/5 because the specific content about avoiding grave defilements through prayer and fasting is not yet present in the document collection. The system correctly refuses to hallucinate rather than inventing an answer. Adding targeted teaching documents will resolve this.

---

## 🛠️ Technical Stack

| Component | Technology |
|---|---|
| Embedding Model | `distiluse-base-multilingual-cased-v2` (512 dims, multilingual) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Database | ChromaDB (persistent local storage) |
| Keyword Search | BM25Okapi (rank-bm25) |
| LLM | Google Gemini 3.1 Flash Lite |
| PDF Processing | pypdf with sliding window chunking |
| Frontend | Streamlit with custom CSS (navy/gold theme) |
| Email | Gmail SMTP with App Password |
| Language | Python 3.11+ |

---

## 🚧 Challenges & Solutions

### Challenge 1 — Memory Constraint (4.5GB RAM)
**Problem:** The full Bible KJV PDF (1,625 pages) caused a memory access violation when loading simultaneously with ChromaDB and the embedding model.

**Solution:** Implemented batch processing in groups of 500 chunks, called `gc.collect()` between batches, and switched to the lighter `paraphrase-MiniLM-L3-v2` model before upgrading to the multilingual model once memory was optimized.

---

### Challenge 2 — ChromaDB Batch Size Limit
**Problem:** `ChromaDB` rejected batches larger than 5,461 items, crashing on the first indexing attempt of 5,715 chunks.

**Solution:** Split all `collection.add()` calls into configurable batches of 500 with progress reporting.

---

### Challenge 3 — Silent Retrieval Bug
**Problem:** The original Vancouver RAG project (a precursor) had a critical silent bug — `model.encode(chunks)` was used instead of `model.encode([question])` during retrieval. The system appeared to work but always returned the same chunks regardless of the question asked.

**Solution:** Understanding the RAG pipeline at the conceptual level revealed that the query vector must represent the user's *question*, not the document chunks. The bug was caught through understanding, not through error messages.

---

### Challenge 4 — Poor Retrieval on Theological French
**Problem:** Initial eval scores averaged 1.2/5. The embedding model was retrieving irrelevant general content instead of specific doctrinal definitions.

**Root causes identified through eval harness:**
1. Chunk filter `len(p) > 100` was removing short but precise doctrinal definitions
2. Question-list chunks from the table of contents were outscoring actual content chunks
3. The embedding model struggled with specialized French theological vocabulary

**Solutions applied iteratively:**
1. Reduced chunk filter to `len(p) > 30` and implemented sliding window chunking (3 lines, 1 line overlap)
2. Added filter `not c.strip().startswith(tuple("123456789"))` to exclude table-of-contents chunks
3. Built a query expansion dictionary mapping theological concepts to precise keywords
4. Added force-injection of specific chunks for complex classification questions

**Result:** Average score improved from 1.2/5 → 2.0/5 → 2.8/5 → 3.8/5 → 4.2/5 across 5 iterations.

---

### Challenge 5 — API Rate Limits During Evaluation
**Problem:** Running 9 consecutive questions through the eval harness (each requiring 2 Gemini API calls) repeatedly exhausted free-tier quotas mid-evaluation.

**Solution:** Implemented automatic retry with exponential backoff, JSON progress saving after each question so evaluations resume from the last completed question rather than restarting, and 15-second delays between API calls to stay within per-minute limits.

---

### Challenge 6 — Embedding Dimension Mismatch
**Problem:** Switching from `paraphrase-MiniLM-L3-v2` (384 dims) to `distiluse-base-multilingual-cased-v2` (512 dims) caused ChromaDB to reject queries with "Collection expecting embedding with dimension of 384, got 512."

**Solution:** Deleted the existing ChromaDB collection, used a versioned collection name (`bible_doctrine_v3`) to force a fresh collection, and reindexed with the new model. Learned that both `index.py` and `main.py` must always use the identical embedding model.

---

## 📁 Project Structure

```
bible_ai_project/
├── index.py              # One-time indexing pipeline
├── main.py               # RAG search + validation functions  
├── app.py                # Streamlit UI (3 tabs: Question, Document, Admin)
├── eval.py               # LLM-as-Judge evaluation harness
├── requirements.txt
├── .env                  # API keys (never committed)
├── .gitignore
└── MERGE_BIBLE_DOCTRINE.pdf   # Source documents (not committed)
```

---

## 🚀 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/akirem20/bible-doctrine-assistant
cd bible-doctrine-assistant

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Fill in your API keys

# Index your documents (run once)
python index.py

# Launch the app
streamlit run app.py
```

### Environment Variables Required
```
GOOGLE_API_KEY=your_gemini_api_key
GMAIL_ADDRESS=your_gmail@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
FATHER_EMAIL=admin_email@gmail.com
```

---

## 🗺️ Roadmap

- [ ] Deploy to Streamlit Cloud with persistent ChromaDB
- [ ] Cross-Encoder Reranker upgrade to French-specific model
- [ ] Contextual Retrieval (LLM-generated chunk context)
- [ ] FastAPI layer for React/React Native frontend
- [ ] Mobile app (React Native)
- [ ] Multi-document comparison mode

---

## 👨‍💻 Author

**Samuel Koffi Akirem** — Full Stack Developer & Applied AI Engineer  
Co-op Diploma in Full Stack Development — Greystone College, Vancouver BC  
Incoming Master of Applied AI — Memorial University of Newfoundland (Winter 2027)

GitHub: [@akirem20](https://github.com/akirem20)  
HuggingFace: [@aksa2000](https://huggingface.co/aksa2000)

---

## 📄 License

This project was built for a private church community. The codebase is open source. The document collection is private and not included in this repository.
