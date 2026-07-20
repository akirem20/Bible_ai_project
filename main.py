import os
import json
import pickle
import smtplib
import threading
import heapq
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pypdf import PdfReader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

PENDING_DOCS_FILE = "documents_attente.json"

# --- Thread-Safe Dynamic Index Manager ---
_index_lock = threading.Lock()
_last_loaded_time = 0.0

# Global caches that threads will safely read from
chunks = []
bm25 = None


def get_indices():
    """Safely retrieves or reloads chunks and BM25 matrix if files change on disk."""
    global chunks, bm25, _last_loaded_time

    chunks_path = "chunks_doctrine.pkl"
    bm25_path = "bm25_doctrine.pkl"

    if not os.path.exists(chunks_path) or not os.path.exists(bm25_path):
        return chunks, bm25

    current_mod_time = os.path.getmtime(chunks_path)

    # Thread lock ensures two sessions don't read/write simultaneously
    with _index_lock:
        if bm25 is None or current_mod_time > _last_loaded_time:
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            with open(bm25_path, "rb") as f:
                bm25 = pickle.load(f)
            _last_loaded_time = current_mod_time

    return chunks, bm25


# Initialize core static models
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
# FIXED: Swapped out English miniLM for a native French semantic reranker
# SWAPPED: Replaced missing repository with a premium native French semantic reranker
model_reranker = CrossEncoder("antoinelouis/cross-encoder-camembert-base-msmarco-fr")

db_client = chromadb.PersistentClient(path="./chroma_doctrine_db")
collection = db_client.get_or_create_collection(name="bible_doctrine_v3")


# ── GESTION DE LA FILE D'ATTENTE JSON ──

def charger_documents_attente():
    if not os.path.exists(PENDING_DOCS_FILE):
        return []
    try:
        with open(PENDING_DOCS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def sauvegarder_document_attente(nom, email, rapport, temp_path, preview=""):
    docs = charger_documents_attente()
    docs = [d for d in docs if d["nom"] != nom]
    docs.append({
        "nom": nom,
        "email": email,
        "rapport": rapport,
        "temp_path": temp_path,
        "preview": preview
    })
    with open(PENDING_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def supprimer_document_attente(nom):
    docs = charger_documents_attente()
    docs = [d for d in docs if d["nom"] != nom]
    with open(PENDING_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


# ── RECHERCHE HYBRIDE LOCAL ──

def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def expand_question(question):
    expansions = {
        "péché": "péché transgression loi Satan mort 1 Jean 3.4",
        "vérité": "vérité Jésus Esprit Amour Évangile",
        "adoration": "adoration Esprit Vérité Jean 4.24 Père adorateurs sanctifier affranchir",
        "lèpre": "lèpre péché agression réaction plaie tumeur cicatrice brûlure orgueil",
        "souillure": "souillure impureté Lévitique eau mort ossements sépulcre biologique",
        "chrétien": "chrétien adoration Esprit Vérité sanctifier affranchir nouvel homme",
        "différents types": "péchés expiable non-expiable Lévitique 4 5 6 20 21 sang eau spéciaux lèpre mort impureté expiation",
        "souillures les plus graves": "graves non-expiables sang meurtres Mammon doctrine mort piété jeûne prière philosophie mondaine Lévitique",
    }
    for keyword, expansion in expansions.items():
        if keyword in question.lower():
            return question + " " + expansion
    return question


def retrieve_context(question):
    # 1. Dynamically pull the latest thread-safe chunks and bm25 index matrix
    current_chunks, current_bm25 = get_indices()

    if not current_chunks or current_bm25 is None:
        return ""

    expanded = expand_question(question)
    normalized_question = expanded.lower().strip()

    # Helper to filter out text chunks starting with numeric noise (e.g., table of contents pages)
    def isValidChunk(c):
        return not c.strip().startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9'))

    # 2. Semantic Search (ChromaDB)
    vec_q = model.encode([normalized_question]).tolist()
    result = collection.query(query_embeddings=vec_q, n_results=10)
    s_search = [c for c in result["documents"][0] if isValidChunk(c)]

    # 3. Lexical Search (BM25) - Optimized with HeapQ to prevent O(N log N) in-memory lists
    h_q = normalized_question.split()
    h_score = current_bm25.get_scores(h_q)
    top_indices_score = heapq.nlargest(10, range(len(h_score)), key=lambda k: h_score[k])
    h_search = [current_chunks[i] for i in top_indices_score if isValidChunk(current_chunks[i])]

    # 4. Hybrid Merge (Deduplicated maintaining ranking order hierarchy)
    combined = list(dict.fromkeys(s_search + h_search))

    # 5. Rule-Based Pre-Rerank Injections (Feeds choices into the CrossEncoder safely before scoring)
    if "différents types" in question.lower():
        expiable_chunks = [c for c in current_chunks if "expiable" in c.lower()][:3]
        combined = list(dict.fromkeys(expiable_chunks + combined))

    if "souillures les plus graves" in question.lower():
        grave_chunks = [
            c for c in current_chunks
            if "souillure" in c.lower() and (
                    "sang" in c.lower() or "mort" in c.lower() or "nombre 19" in c.lower() or "purifi" in c.lower())
        ][:5]
        combined = list(dict.fromkeys(grave_chunks + combined))

    # Cap candidate pool to 20 before executing deep neural cross-attention matrix scoring
    combined = combined[:20]

    if not combined:
        return ""

    # 6. Cross-Encoder Reranking (Using full expanded string for better semantic evaluation context)
    pairs = [[expanded, chunk] for chunk in combined]
    cross_scores = model_reranker.predict(pairs)

    # Safely select top 5 highest-relevance ranked chunks
    top_indice_cross = heapq.nlargest(5, range(len(cross_scores)), key=lambda k: cross_scores[k])
    final_chunks = [combined[i] for i in top_indice_cross]

    return "\n".join(final_chunks)


# ── CORRECTION DU PROMPT : EXIGER LE TEXTE DE LA BIBLE EN TOUTES LETTRES ──

def ai_search(question):
    context = retrieve_context(question)

    prompt = f"""
    Utilisez le contexte fourni et vos connaissances théologiques pour répondre précisément à la question.
    Répondez STRICTEMENT en français.

    RÈGLE CRITIQUE ET ABSOLUE POUR LES PASSAGES BIBLIQUES :
    Pour CHAQUE verset ou passage biblique que vous mentionnez ou utilisez pour étayer votre réponse, vous devez OBLIGATOIREMENT écrire son TEXTE INTÉGRAL ET MOT À MOT en français. Ne donnez JAMAIS uniquement les numéros de chapitres ou versets seuls (par exemple, n'écrivez pas juste "Ép 3:8-9", vous devez écrire "Éphésiens 3:8-9 : [Insérer ici le texte complet et mot à mot du verset]"). L'utilisateur doit pouvoir lire la Parole de Dieu écrite en entier au début de la réponse.

    RÈGLES STRICTES DE FORMATAGE :
    - N'utilisez AUCUN astérisque (* ou **) nulle part dans le texte.
    - N'utilisez aucun tiret ou puce de liste markdown.
    - Écrivez chaque verset complet sur sa propre ligne isolée.
    - Séparez clairement les citations bibliques initiales de votre explication doctrinale finale par un saut de ligne.

    Si les éléments de réponse ne figurent pas du tout dans le contexte, indiquez poliment que vous ne pouvez pas répondre.

    Contexte : {context}
    Question : {question}
    """

    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Vous êtes un Expert Théologien dogmatique. Vous écrivez exclusivement en texte brut "
                    "sans aucun symbole markdown (pas d'astérisques). Vous devez obligatoirement citer "
                    "le texte intégral des versets de la Bible que vous invoquez."
                ),
                temperature=0.0,
            ),
        )
        return response.text
    except errors.APIError as e:
        err_msg = getattr(e, 'message', str(e))
        err_code = getattr(e, 'code', getattr(e, 'status_code', 'unknown'))
        return f"⚠️ Erreur API Gemini (Status {err_code}) : {err_msg}"


def extract_claims(text):
    prompt = f"""
    Lisez le document théologique suivant et extrayez les principales affirmations doctrinales qu'il contient.
    Retournez-les sous forme de liste numérotée en français. Maximum 10 affirmations.
    Soyez concis et précis.

    Document:
    {text}
    """
    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Vous êtes un théologien biblique expert. Votre rôle est d'extraire les affirmations "
                    "doctrinales. Vous devez rédiger votre réponse exclusivement en français."
                ),
                temperature=0.0,
            ),
        )
        return response.text
    except errors.APIError as e:
        err_msg = getattr(e, 'message', str(e))
        err_code = getattr(e, 'code', getattr(e, 'status_code', 'unknown'))
        raise RuntimeError(f"Erreur API Gemini lors de l'extraction ({err_code}): {err_msg}") from e


def validate_document(path):
    """Génère un rapport d'évaluation complet en français avec une note claire sur 5."""
    try:
        newdoc = PdfReader(path)
        text_parts = []
        for page in newdoc.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        text = "\n".join(text_parts).strip()

        if not text:
            return "⚠️ Erreur : Le document PDF ne contient pas de texte extractible."

        if len(text) > 50000:
            text = text[:50000] + "\n\n[... Texte tronqué pour l'analyse ...]"

        try:
            ext_claims = extract_claims(text)
        except Exception as e:
            return f"⚠️ Erreur lors du extraction des affirmations : {str(e)}"

        claims_list = []
        for line in ext_claims.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                cleaned = line.lstrip("0123456789.-* ")
                if cleaned:
                    claims_list.append(cleaned)

        if not claims_list:
            claims_list = [line.strip() for line in ext_claims.split("\n") if line.strip()][:5]

        matched_doctrinal_context = []
        for claim in claims_list[:5]:
            db_context = retrieve_context(claim)
            matched_doctrinal_context.append(f"Pour l'affirmation: '{claim}'\nDoctrine établie:\n{db_context}")

        existing_doctrine = "\n\n---\n\n".join(matched_doctrinal_context)

        prompt = f"""
        Comparez les affirmations du nouveau document avec la doctrine existante.
        Attribuez obligatoirement une Note globale sous le format fractionnaire strict sur 5 (ex: 1/5, 3/5, 5/5).

        Rédigez TOUT votre rapport en FRANÇAIS selon cette structure exacte :
        Note globale : X/5

        Analyse détaillée des points :
        (Détaillez ici point par point en français la conformité ou les erreurs trouvées)

        Affirmations extraites :
        {ext_claims}

        Doctrine de référence :
        {existing_doctrine}
        """

        client = get_genai_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Agissez en tant que théologien expert. Rédigez tout votre rapport en français. "
                    "N'utilisez aucune phrase en anglais. La note doit impérativement être écrite sous la forme X/5."
                ),
                temperature=0.0,
            ),
        )
        return response.text

    except errors.APIError as e:
        err_msg = getattr(e, 'message', str(e))
        err_code = getattr(e, 'code', getattr(e, 'status_code', 'unknown'))
        return f"⚠️ Erreur API Gemini (Status {err_code}) : {err_msg}"
    except Exception as e:
        return f"⚠️ Erreur lors du traitement : {str(e)}"


def send_notification_email(document_name, validation_report):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    receiver = os.getenv("FATHER_EMAIL")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = f"Nouveau document en attente de validation : {document_name}"

    body = f"Bonjour,\n\nUn document a été soumis.\n\nDocument : {document_name}\n\nRapport :\n{validation_report}"
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())


def add_to_collection(path):
    # Retrieve current active indices within thread lock context
    current_chunks, _ = get_indices()

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
        chunk = " ".join(lines[i: i + 3])
        if len(chunk) > 50:
            new_chunks.append(chunk)

    start_id = len(current_chunks)
    new_ids = [f"id_{start_id + i}" for i in range(len(new_chunks))]

    batch_size = 256
    for i in range(0, len(new_chunks), batch_size):
        batch_chunks = new_chunks[i: i + batch_size]
        batch_ids = new_ids[i: i + batch_size]
        batch_embeddings = model.encode(batch_chunks).tolist()
        collection.add(embeddings=batch_embeddings, ids=batch_ids, documents=batch_chunks)

    # Use the isolated thread lock to cleanly update files on disk
    with _index_lock:
        # Re-read fresh right before extending to avoid dropping mid-flight edits from other sessions
        with open("chunks_doctrine.pkl", "rb") as f:
            fresh_chunks = pickle.load(f)

        fresh_chunks.extend(new_chunks)
        tokenized_chunks = [chunk.lower().split() for chunk in fresh_chunks]
        new_bm25 = BM25Okapi(tokenized_chunks)

        with open("chunks_doctrine.pkl", "wb") as pkl_file:
            pickle.dump(fresh_chunks, pkl_file)
        with open("bm25_doctrine.pkl", "wb") as pkl_file:
            pickle.dump(new_bm25, pkl_file)

    return len(new_chunks)