import streamlit as st
import os
import zipfile
import requests
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="Assistant Doctrinal",
    page_icon="✝️",
    layout="wide"
)

# 2. Database Bootstrap Constants
DB_ZIP = "doctrine_db.zip"
DB_FOLDER = "chroma_doctrine_db"
CHUNKS_FILE = "chunks_doctrine.pkl"
BM25_FILE = "bm25_doctrine.pkl"
DB_DOWNLOAD_URL = "https://github.com/akirem20/Bible_ai_project/releases/download/v1.0.0/doctrine_db.zip"


@st.cache_resource
def bootstrap_database():
    """Télécharge et extrait les fichiers de la base de données s'ils sont manquants sur le serveur Streamlit."""
    if not (os.path.exists(DB_FOLDER) and os.path.exists(CHUNKS_FILE) and os.path.exists(BM25_FILE)):
        with st.spinner("⚡ Configuration de l'application : Restauration de la base de données doctrinale..."):
            headers = {}
            if "GITHUB_TOKEN" in st.secrets:
                headers["Authorization"] = f"token {st.secrets['GITHUB_TOKEN']}"

            response = requests.get(DB_DOWNLOAD_URL, headers=headers, stream=True)
            if response.status_code == 200:
                with open(DB_ZIP, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                with zipfile.ZipFile(DB_ZIP, 'r') as zip_ref:
                    zip_ref.extractall(".")

                os.remove(DB_ZIP)
            else:
                st.error(f"Échec du téléchargement de la base de données. Code statut: {response.status_code}")
                st.stop()


bootstrap_database()

from main import (
    ai_search,
    validate_document,
    add_to_collection,
    send_notification_email,
    sauvegarder_document_attente,
    charger_documents_attente,
    supprimer_document_attente
)

load_dotenv()

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght=400;700;900&family=Inter:wght=300;400;500&display=swap');

    .stApp { background-color: #080d1a; color: #f0ece0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-section { padding: 4rem 2rem 2rem 2rem; }
    .input-section { padding: 4rem 2rem 2rem 2rem; }

    .eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #c9a84c;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .eyebrow::before {
        content: '';
        display: inline-block;
        width: 2rem;
        height: 1px;
        background: #c9a84c;
    }

    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(2.2rem, 4vw, 3.5rem);
        font-weight: 900;
        line-height: 1.1;
        color: #f0ece0;
        margin-bottom: 1.2rem;
        opacity: 1;
    }
    .hero-title .gold-word { color: #c9a84c; }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #8896a7;
        line-height: 1.8;
        margin-bottom: 2rem;
    }

    .section-divider { border: none; border-top: 1px solid #1a2540; margin: 2rem 0; }

    .features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; padding: 2rem 0; }
    .feature-card {
        background: #0f1828;
        border: 1px solid #1a2540;
        border-radius: 6px;
        padding: 1.5rem;
        transition: border-color 0.3s ease;
    }
    .feature-card:hover { border-color: #c9a84c; }
    .feature-icon { font-size: 1.3rem; margin-bottom: 0.75rem; }
    .feature-title { font-family: 'Playfair Display', Georgia, serif; font-size: 1rem; color: #f0ece0; margin-bottom: 0.5rem; }
    .feature-desc { font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #8896a7; line-height: 1.7; }

    .bottom-bar {
        border-top: 1px solid #1a2540;
        padding: 1.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #8896a7;
        text-align: center;
        letter-spacing: 0.05em;
        margin-top: 2rem;
    }

    .stTextInput > div > div > input {
        background-color: #0f1828 !important;
        color: #f0ece0 !important;
        border: 1px solid #1a2540 !important;
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput > div > div > input:focus { border-color: #c9a84c !important; }
    .stButton > button {
        background: #c9a84c !important;
        color: #080d1a !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover { background: #b8943d !important; }
    .stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1a2540 !important; }
    .stTabs [data-baseweb="tab"] { color: #8896a7 !important; font-family: 'Inter', sans-serif !important; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; }
    .stTabs [aria-selected="true"] { color: #c9a84c !important; border-bottom: 2px solid #c9a84c !important; }

    .answer-card {
        background: #0f1828;
        border-left: 3px solid #c9a84c;
        border-radius: 4px;
        padding: 2rem;
        margin-top: 1.5rem;
        font-family: 'Inter', sans-serif;
        color: #d4cfc4;
        line-height: 1.9;
    }

    /* Admin Shared Unified Container Card style */
    .admin-report-card {
        background-color: #0f1828;
        border: 1px solid #1a2540;
        border-radius: 6px;
        padding: 1.75rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        color: #f0ece0;
        font-family: 'Inter', sans-serif;
    }

    @media (max-width: 768px) {
        .hero-section { padding: 2rem 1rem 1rem 1rem !important; }
        .input-section { padding: 1.5rem 1rem 1.5rem 1rem !important; }
        .features { grid-template-columns: 1fr !important; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT
# ============================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="hero-section">
        <div class="eyebrow">Assistant Biblique IA</div>
        <h1 class="hero-title">
            Discernement<br>
            doctrinal, avec<br>
            une réponse <span class="gold-word">claire.</span>
        </h1>
        <p class="hero-subtitle">
            Posez vos questions sur la doctrine biblique,
            comparez les réponses avec les Écritures, et
            gardez une trace des sources approuvées.
        </p>
        <hr class="section-divider">
        <div class="features">
            <div class="feature-card">
                <div class="feature-icon">📖</div>
                <div class="feature-title">Écritures d'abord</div>
                <p class="feature-desc">Chaque réponse commence par les passages bibliques pertinents.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">✅</div>
                <div class="feature-title">Études approuvées</div>
                <p class="feature-desc">Comparées aux notes doctrinales validées par votre équipe.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Documents utiles</div>
                <p class="feature-desc">Soumettez notes, leçons ou extraits pour validation.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="input-section">
        <div style="font-family: Inter, sans-serif; font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: #c9a84c; margin-bottom: 0.5rem;">
            Question doctrinale
        </div>
        <div style="font-family: Playfair Display, Georgia, serif; font-size: 1.4rem; color: #f0ece0; margin-bottom: 0.5rem;">
            Commencez ici
        </div>
    </div>
    """, unsafe_allow_html=True)

    is_admin = st.query_params.get("admin", "") == "true"

    if is_admin:
        tab1, tab2, tab3 = st.tabs(["Question", "Document", "Admin"])
    else:
        tab1, tab2 = st.tabs(["Question", "Document"])
        tab3 = None

    if "doc_success_msg" in st.session_state:
        st.success(st.session_state["doc_success_msg"])
        del st.session_state["doc_success_msg"]

    # ── TAB 1 — QUESTION ──
    with tab1:
        question = st.text_input("Votre question", placeholder="Ex: Quel est le mystère de Dieu ?",
                                 key="input_question")
        if st.button("Rechercher", key="search"):
            if question:
                with st.spinner("Recherche dans les Écritures..."):
                    answer = ai_search(question)
                answer_html = answer.replace("\n", "<br>")
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-label">✝ Réponse doctrinale</div>
                    <hr style="border:none; border-top:1px solid #1a2540; margin:1rem 0 1.5rem 0;">
                    {answer_html}
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2 — DOCUMENT (USER SIDE) ──
    with tab2:
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#8896a7;margin-bottom:1rem;'>Le document sera analysé et envoyé au responsable pour approbation.</p>",
            unsafe_allow_html=True)

        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0

        uploaded_file = st.file_uploader("Choisir un fichier PDF", type=["pdf"],
                                         key=f"pdf_uploader_{st.session_state['uploader_key']}")
        submitter_email = st.text_input("Votre adresse email", placeholder="votre@email.com", key="input_email")

        if st.button("Soumettre pour validation", key="submit"):
            if uploaded_file and submitter_email:
                with st.spinner("Traitement en cours..."):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # Extraction locale de l'aperçu du sujet en Français
                    try:
                        from pypdf import PdfReader

                        reader = PdfReader(temp_path)
                        extracted_text = ""
                        for page in reader.pages[:2]:
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text += page_text + "\n"
                        preview_content = extracted_text.strip()[:800]
                        if not preview_content:
                            preview_content = "Contenu textuel brut introuvable (Document potentiellement scanné)."
                    except Exception as e:
                        preview_content = f"Erreur de lecture de l'aperçu : {str(e)}"

                    # Génération du rapport de validation pur français
                    raw_report = validate_document(temp_path)
                    validation_report = raw_report.replace("**", "")

                    sauvegarder_document_attente(
                        uploaded_file.name,
                        submitter_email,
                        validation_report,
                        temp_path,
                        preview_content
                    )

                try:
                    send_notification_email(uploaded_file.name, validation_report)
                except Exception:
                    pass

                st.session_state["doc_success_msg"] = "Document soumis avec succès. Le responsable a été notifié."
                st.session_state["uploader_key"] += 1
                st.rerun()

    # ── TAB 3 — ADMIN (ADMIN SIDE) ──
    if tab3 is not None:
        with tab3:
            if "admin_logged_in" not in st.session_state:
                st.session_state["admin_logged_in"] = False

            if not st.session_state["admin_logged_in"]:
                if st.button("Envoyer un code de connexion", key="send_code"):
                    code = str(random.randint(100000, 999999))
                    st.session_state["login_code"] = code
                    sender = os.getenv("GMAIL_ADDRESS")
                    password = os.getenv("GMAIL_APP_PASSWORD")
                    receiver = os.getenv("FATHER_EMAIL")
                    msg = MIMEMultipart()
                    msg["From"] = sender
                    msg["To"] = receiver
                    msg["Subject"] = "Code de connexion — Admin"
                    msg.attach(MIMEText(f"Votre code : {code}", "plain"))
                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                        server.starttls()
                        server.login(sender, password)
                        server.sendmail(sender, receiver, msg.as_string())
                    st.success("Code envoyé.")

                entered_code = st.text_input("Code reçu", type="password", key="admin_code_input")
                if st.button("Se connecter", key="login"):
                    if "login_code" in st.session_state and entered_code == st.session_state["login_code"]:
                        st.session_state["admin_logged_in"] = True
                        st.rerun()

            else:
                col_h, col_l = st.columns([4, 1])
                with col_h:
                    st.markdown("### Documents en attente de validation")
                with col_l:
                    if st.button("Déconnexion", key="logout"):
                        st.session_state["admin_logged_in"] = False
                        st.rerun()

                docs_a_valider = charger_documents_attente()

                if not docs_a_valider:
                    st.info("Aucun document en attente.")
                else:
                    for index, doc in enumerate(docs_a_valider):
                        st.markdown(f"📄 <b>Fichier :</b> {doc['nom']}", unsafe_allow_html=True)
                        st.markdown(f"📧 <b>Soumis par :</b> {doc['email']}", unsafe_allow_html=True)

                        # Conversion des retours à la ligne pour l'affichage HTML propre
                        sujet_html = doc.get("preview", "Aucun aperçu disponible.").replace("\n", "<br>")
                        rapport_html = doc['rapport'].replace("\n", "<br>")

                        # Boîte unique fusionnée — Fond natif du site avec texte blanc clair
                        st.markdown(f"""
                        <div class="admin-report-card">
                            <span style="color: #c9a84c; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase;">📝 Contenu / Sujet du document</span>
                            <p style="color: #d4cfc4; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 1.5rem; line-height: 1.6;">
                                {sujet_html}
                            </p>

                            <hr style="border: none; border-top: 1px solid #1a2540; margin: 1.5rem 0;">

                            <span style="color: #c9a84c; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase;">📊 Rapport d'Analyse Théologique</span>
                            <p style="color: #f0ece0; font-size: 0.95rem; margin-top: 0.5rem; line-height: 1.7;">
                                {rapport_html}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Approuver", key=f"approve_{index}"):
                                if os.path.exists(doc['temp_path']):
                                    add_to_collection(doc['temp_path'])
                                    os.remove(doc['temp_path'])
                                supprimer_document_attente(doc['nom'])
                                st.rerun()
                        with col2:
                            if st.button("❌ Rejeter", key=f"reject_{index}"):
                                if os.path.exists(doc['temp_path']):
                                    os.remove(doc['temp_path'])
                                supprimer_document_attente(doc['nom'])
                                st.rerun()

                        st.markdown("<hr style='border-top: 1px dashed #1a2540; margin: 2rem 0;'>",
                                    unsafe_allow_html=True)

# ============================================================
# BOTTOM BAR
# ============================================================
st.markdown('<div class="bottom-bar">Assistant Doctrinal — Gestion des Écritures</div>', unsafe_allow_html=True)