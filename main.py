import os
import json
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
from google.genai import errors
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

# Fichier de persistance local pour les documents en attente
PENDING_DOCS_FILE = "documents_attente.json"

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


# ── FONCTIONS DE GESTION DE LA FILE D'ATTENTE (JSON) ──

def charger_documents_attente():
    """Charge la liste des documents en attente depuis le fichier JSON local."""
    if not os.path.exists(PENDING_DOCS_FILE):
        return []
    try:
        with open(PENDING_DOCS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def sauvegarder_document_attente(nom, email, rapport, temp_path, preview=""):
    """Enregistre un nouveau document soumis dans le fichier JSON local avec son aperçu textuel."""
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
    """Supprime un document de la file d'attente JSON après traitement (Approbation/Rejet)."""
    docs = charger_documents_attente()
    docs = [d for d in docs if d["nom"] != nom]

    with open(PENDING_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


# ── FONCTIONS RAG ET THÉOLOGIQUES EXISTANTES ──

def get_genai_client():
    """Safely retrieves the Gemini API Key from all possible environments."""
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
    """Local hybrid search (Semantic + BM25) to extract relevant database text."""
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

    if "différents types" in question.lower():
        expiable_chunks = [c for c in chunks if "expiable" in c.lower()][:3]
        combined = list(dict.fromkeys(expiable_chunks + combined))[:20]

    if "souillures les plus graves" in question.lower():
        grave_chunks = [
            c for c in chunks
            if "souillure" in c.lower() and (
                        "sang" in c.lower() or "mort" in c.lower() or "nombre 19" in c.lower() or "purifi" in c.lower())
        ][:5]
        combined = list(dict.fromkeys(grave_chunks + combined))[:20]

    # Cross-Encoder reranking
    pairs = [[question, chunk] for chunk in combined]
    cross_scores = model_reranker.predict(pairs)
    top_indice_cross = sorted(
        range(len(cross_scores)), key=lambda k: cross_scores[k], reverse=True
    )[:5]
    final_chunks = [combined[i] for i in top_indice_cross]

    if "souillures les plus graves" in question.lower():
        grave_chunks = [
            c for c in chunks
            if "souillure" in c.lower() and (
                        "sang" in c.lower() or "mort" in c.lower() or "nombre 19" in c.lower() or "purifi" in c.lower())
        ][:3]
        final_chunks = list(dict.fromkeys(grave_chunks + final_chunks))[:5]

    return "\n".join(final_chunks)


def ai_search(question):
    """User-facing search."""
    context = retrieve_context(question)

    prompt = f"""
    Only use the context and the question to answer the user question.
    Answer strictly in French.
    Include relevant Bible verses first then answer.

    CRITICAL FORMATTING RULES:
    - Do NOT use any asterisks (* or **) anywhere in your response.
    - Do not use markdown bullet points. 
    - To list Bible verses, simply start a new line for each verse (e.g., "Romains 5:12 : [Text]").
    - Use clean, normal paragraph breaks for your explanation.

    If the answer is not in the context say you are not able to answer.

    Context: {context}

    Question: {question}
    """

    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Act as a Theologian Expert to guide users based on church"
                    " doctrine. Always write in plain text without markdown formatting."
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
    """Extracts claims ensuring output is generated 100% in French."""
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
    """Generates comparison report strictly in French with fractionary grading style (X/5)."""
    try:
        newdoc = PdfReader(path)
        text_parts = []
        for page in newdoc.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        text = "\n".join(text_parts).strip()

        if not text:
            return (
                "⚠️ Erreur : Le document PDF ne contient pas de texte extractible. "
                "Veuillez utiliser un fichier PDF textuel classique."
            )

        if len(text) > 50000:
            text = text[:50000] + "\n\n[... Texte tronqué pour l'analyse ...]"

        try:
            ext_claims = extract_claims(text)
        except Exception as e:
            return f"⚠️ Erreur lors de l'extraction des affirmations : {str(e)}"

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
            matched_doctrinal_context.append(
                f"Pour l'affirmation: '{claim}'\nDoctrine établie correspondante:\n{db_context}"
            )

        existing_doctrine = "\n\n---\n\n".join(matched_doctrinal_context)

        prompt = f"""
        Comparez les affirmations du nouveau document avec la doctrine existante de l'église.
        Attribuez une note globale d'alignement sous format fractionnaire obligatoire sur 5 (par exemple: 1/5, 3/5 ou 5/5), 
        où 1/5 signifie une forte contradiction et 5/5 signifie un alignement parfait.

        Rédigez obligatoirement l'intégralité de votre réponse en FRANÇAIS en respectant exactement cette structure :
        Note globale : X/5

        Analyse détaillée :
        (Détaillez ici en français l'accord ou la contradiction pour chaque point extrait)

        Affirmations du nouveau document :
        {ext_claims}

        Doctrine de référence retrouvée en base :
        {existing_doctrine}
        """

        client = get_genai_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Agissez en tant qu'expert théologien biblique comparant les nouveaux enseignements "
                    "à la doctrine établie. Rédigez tout votre rapport en français. Ne laissez aucun texte en anglais."
                ),
                temperature=0.0,
            ),
        )
        return response.text

    except errors.APIError as e:
        err_msg = getattr(e, 'message', str(e))
        err_code = getattr(e, 'code', getattr(e, 'status_code', 'unknown'))
        return f"⚠️ Erreur API Gemini lors de la validation (Status {err_code}) : {err_msg}"
    except Exception as e:
        return f"⚠️ Erreur inattendue lors du traitement du document : {str(e)}"


def send_notification_email(document_name, validation_report):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    receiver = os.getenv("FATHER_EMAIL")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = f"Nouveau document en attente de validation : {document_name}"

    body = f"""
Bonjour,
Un nouveau document a été soumis pour validation doctrinale.
Document : {document_name}

Rapport de validation :
{validation_report}

Veuillez vous connecter à l'application pour valider ce document.
Cordialement,
L'Assistant Doctrinal
    """
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())


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
        chunk = " ".join(lines[i: i + 3])
        if len(chunk) > 50:
            new_chunks.append(chunk)

    start_id = len(chunks)
    new_ids = [f"id_{start_id + i}" for i in range(len(new_chunks))]

    batch_size = 256
    for i in range(0, len(new_chunks), batch_size):
        batch_chunks = new_chunks[i: i + batch_size]
        batch_ids = new_ids[i: i + batch_size]
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

    return len(new_chunks)


if __name__ == "__main__":
    pass