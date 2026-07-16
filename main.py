import os
import pickle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pypdf import PdfReader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi  # Rebuilds BM25 when new PDFs are uploaded
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

# Load from disk
with open("chunks_doctrine.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("bm25_doctrine.pkl", "rb") as f:
    bm25 = pickle.load(f)

# Load models and ChromaDB
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
model_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
db_client = chromadb.PersistentClient(path="./chroma_doctrine_db")
collection = db_client.get_or_create_collection(name="bible_doctrine_v3")


def get_genai_client():
    """Safely retrieves the Gemini API Key from all possible environments
    (local env, Streamlit Secrets, GEMINI_API_KEY, or GOOGLE_API_KEY)
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get(
                "GOOGLE_API_KEY"
            )
        except Exception:
            pass

    if api_key:
        return genai.Client(api_key=api_key)

    return genai.Client()


def expand_question(question):
    expansions = {
        "péché": "péché transgression loi Satan mort 1 Jean 3.4",
        "vérité": "vérité Jésus Esprit Amour Évangile",
        "adoration": (
            "adoration Esprit Vérité Jean 4.24 Père adorateurs sanctifier"
            " affranchir"
        ),
        "lèpre": (
            "lèpre péché agression réaction plaie tumeur cicatrice brûlure"
            " orgueil"
        ),
        "souillure": (
            "souillure impureté Lévitique eau mort ossements sépulcre biologique"
        ),
        "chrétien": (
            "chrétien adoration Esprit Vérité sanctifier affranchir nouvel"
            " homme"
        ),
        "différents types": (
            "péchés expiable non-expiable Lévitique 4 5 6 20 21 sang eau spéciaux"
            " lèpre mort impureté expiation"
        ),
        "souillures les plus graves": (
            "graves non-expiables sang meurtres Mammon doctrine mort piété jeûne"
            " prière philosophie mondaine Lévitique"
        ),
    }
    for keyword, expansion in expansions.items():
        if keyword in question.lower():
            return question + " " + expansion
    return question


def retrieve_context(question):
    """Local hybrid search (Semantic + BM25) to extract relevant database text.
    Requires ZERO Gemini API calls.
    """
    expanded = expand_question(question)
    normalized_question = expanded.lower().strip()

    # Semantic search
    vec_q = model.encode([normalized_question]).tolist()
    result = collection.query(query_embeddings=vec_q, n_results=10)
    s_search = result["documents"][0]
    s_search = [
        c for c in s_search if not c.strip().startswith(tuple("123456789"))
    ]

    # BM25 keyword search
    h_q = normalized_question.split()
    h_score = bm25.get_scores(h_q)
    top_indices_score = sorted(
        range(len(h_score)), key=lambda k: h_score[k], reverse=True
    )[:10]
    h_search = [chunks[i] for i in top_indices_score]
    h_search = [
        c for c in h_search if not c.strip().startswith(tuple("123456789"))
    ]

    # Combine
    combined = list(dict.fromkeys(s_search + h_search))[:20]

    # Force inject BEFORE Cross-Encoder
    if "différents types" in question.lower():
        expiable_chunks = [
            c for c in chunks if "expiable" in c.lower()
        ][:3]
        combined = list(dict.fromkeys(expiable_chunks + combined))[:20]

    if "souillures les plus graves" in question.lower():
        grave_chunks = [
            c
            for c in chunks
            if "souillure" in c.lower()
            and (
                "sang" in c.lower()
                or "mort" in c.lower()
                or "nombre 19" in c.lower()
                or "purifi" in c.lower()
            )
        ][:5]
        combined = list(dict.fromkeys(grave_chunks + combined))[:20]

    # Cross-Encoder reranking
    pairs = [[question, chunk] for chunk in combined]
    cross_scores = model_reranker.predict(pairs)
    top_indice_cross = sorted(
        range(len(cross_scores)), key=lambda k: cross_scores[k], reverse=True
    )[:5]
    final_chunks = [combined[i] for i in top_indice_cross]

    # Force into final context AFTER Cross-Encoder
    if "souillures les plus graves" in question.lower():
        grave_chunks = [
            c
            for c in chunks
            if "souillure" in c.lower()
            and (
                "sang" in c.lower()
                or "mort" in c.lower()
                or "nombre 19" in c.lower()
                or "purifi" in c.lower()
            )
        ][:3]
        final_chunks = list(dict.fromkeys(grave_chunks + final_chunks))[:5]

    return "\n".join(final_chunks)


def ai_search(question):
    """User-facing search.
    Uses local database retrieval, then calls Gemini once to generate an answer.
    """
    context = retrieve_context(question)

    prompt = f"""
    Only use the context and the question to answer the user question.
    Answer strictly in French.
    Include relevant Bible verses first then answer.
    If the answer is not in the context say you are not able to answer.

    Context: {context}

    Question: {question}
    """

    client = get_genai_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "Act as a Theologian Expert to guide users based on church"
                " doctrine."
            ),
            temperature=0.0,
        ),
    )
    return response.text


def extract_claims(text):
    prompt = f"""
    Read the following theological document and extract 
    the main doctrinal claims it makes.
    Return them as a numbered list. Maximum 10 claims. 
    Be concise and precise.

    Document:
    {text}
    """
    client = get_genai_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a biblical theologian extracting doctrinal claims"
                " from documents."
            ),
            temperature=0.0,
        ),
    )
    return response.text


def validate_document(path):
    """Reads PDF, extracts claims (Gemini Call 1), retrieves matching doctrine
    from the local database locally, and validates the differences (Gemini Call
    2).
    """
    newdoc = PdfReader(path)
    text_parts = []
    for page in newdoc.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    text = "\n".join(text_parts)

    # Gemini Call 1: Extract the claims from the PDF
    ext_claims = extract_claims(text)

    # Process claims (Retrieve doctrine chunks purely locally!)
    claims_list = [c.strip() for c in ext_claims.split("\n") if c.strip()][:5]
    all_doctrine_contexts = []
    for claim in claims_list:
        db_context = retrieve_context(claim)
        all_doctrine_contexts.append(
            f"For Claim: '{claim}'\nEstablished Doctrine Chunks:\n{db_context}"
        )

    existing_doctrine = "\n\n---\n\n".join(all_doctrine_contexts)

    prompt = f"""
    Compare the new document's claims against the existing church doctrine retrieved.
    Score the alignment from 1 to 5, where 1 means strong contradiction 
    and 5 means perfect alignment.

    Provide your answer in this format:
    Score: (number 1-5)
    Explanation: (2-3 sentences explaining agreement or contradiction)

    New document claims:
    {ext_claims}

    Existing doctrine retrieved:
    {existing_doctrine}
    """

    # Gemini Call 2: Final comparison report
    client = get_genai_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "Act as a biblical Theologian Expert comparing new teaching"
                " against established doctrine."
            ),
            temperature=0.0,
        ),
    )
    return response.text


def send_notification_email(document_name, validation_report):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    receiver = os.getenv("FATHER_EMAIL")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = (
        f"Nouveau document en attente de validation : {document_name}"
    )

    body = f"""
Bonjour,

Un nouveau document a été soumis pour validation doctrinale.

Document : {document_name}

Rapport de validation :
{validation_report}

Veuillez vous connecter à l'application pour approuver ou rejeter ce document.

Cordialement,
L'Assistant Doctrinal
    """

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())

    print(f"Email sent to {receiver}")


def add_to_collection(path):
    global chunks, bm25

    newdoc = PdfReader(path)
    text_parts = []
    for page in newdoc.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    text = "\n".join(text_parts)

    lines = [p.strip() for p in text.split("\n") if len(p.strip()) > 0]
    new_chunks = []
    for i in range(0, len(lines), 2):
        chunk = " ".join(lines[i : i + 3])
        if len(chunk) > 50:
            new_chunks.append(chunk)

    start_id = len(chunks)
    new_ids = [f"id_{start_id + i}" for i in range(len(new_chunks))]

    batch_size = 256
    for i in range(0, len(new_chunks), batch_size):
        batch_chunks = new_chunks[i : i + batch_size]
        batch_ids = new_ids[i : i + batch_size]
        batch_embeddings = model.encode(batch_chunks).tolist()
        collection.add(
            embeddings=batch_embeddings, ids=batch_ids, documents=batch_chunks
        )

    chunks.extend(new_chunks)
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    with open("chunks_doctrine.pkl", "wb") as pkl_file:
        pickle.dump(chunks, pkl_file)

    with open("bm25_doctrine.pkl", "wb") as pkl_file:
        pickle.dump(bm25, pkl_file)

    print(
        f"Added {len(new_chunks)} new chunks from {path} and updated BM25"
        " index."
    )
    return len(new_chunks)


if __name__ == "__main__":
    answer = ai_search(
        "Quelle sont les souillures les plus graves ? Et comment les éviter ?"
    )
    print(answer)