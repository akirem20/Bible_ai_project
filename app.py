import streamlit as st
import os
import zipfile
import requests
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# 1. Streamlit Page Configuration (Must be the very first Streamlit command)
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
        with st.spinner(
                "⚡ Configuration de l'application : Restauration de la base de données doctrinale (~10 secondes)..."):
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


# 3. Run the bootstrap BEFORE importing local RAG modules to prevent FileNotFoundError
bootstrap_database()

# 4. Import local assets and standard library dependencies safely
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
# CUSTOM CSS STYLING (Now 100% Responsive for Mobile & Desktop)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght=400;700;900&family=Inter:wght=300;400;500&display=swap');

    .stApp { background-color: #080d1a; color: #f0ece0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Wrapper sections to handle responsive paddings */
    .hero-section {
        padding: 4rem 2rem 2rem 2rem;
    }
    .input-section {
        padding: 4rem 2rem 2rem 2rem;
    }

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
        animation: fadeSlideUp 0.8s ease forwards;
        opacity: 0;
    }
    .hero-title .gold-word { color: #c9a84c; }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #8896a7;
        line-height: 1.8;
        margin-bottom: 2rem;
        animation: fadeSlideUp 0.8s ease 0.2s forwards;
        opacity: 0;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #1a2540;
        margin: 2rem 0;
    }

    /* Grid configuration for desktop */
    .features {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        padding: 2rem 0;
    }
    .feature-card {
        background: #0f1828;
        border: 1px solid #1a2540;
        border-radius: 6px;
        padding: 1.5rem;
        transition: border-color 0.3s ease;
    }
    .feature-card:hover { border-color: #c9a84c; }
    .feature-icon { font-size: 1.3rem; margin-bottom: 0.75rem; }
    .feature-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1rem;
        color: #f0ece0;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #8896a7;
        line-height: 1.7;
    }

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

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Streamlit overrides */
    .stTextInput > div > div > input {
        background-color: #0f1828 !important;
        color: #f0ece0 !important;
        border: 1px solid #1a2540 !important;
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #c9a84c !important;
        box-shadow: 0 0 0 1px #c9a84c !important;
    }
    .stTextInput label {
        color: #8896a7 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
    }
    .stButton > button {
        background: #c9a84c !important;
        color: #080d1a !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: #b8943d !important;
        transform: translateY(-1px) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1a2540 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #8896a7 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }
    .stTabs [aria-selected="true"] {
        color: #c9a84c !important;
        border-bottom: 2px solid #c9a84c !important;
    }
    .answer-card {
        background: #0f1828;
        border-left: 3px solid #c9a84c;
        border-radius: 4px;
        padding: 2rem;
        margin-top: 1.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.9;
        color: #d4cfc4;
        animation: fadeSlideUp 0.5s ease forwards;
    }
    .answer-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #c9a84c;
        margin-bottom: 1rem;
    }
    .gold-divider {
        border: none;
        border-top: 1px solid #1a2540;
        margin: 1rem 0 1.5rem 0;
    }

    @media (max-width: 768px) {
        .hero-section { padding: 2rem 1rem 1rem 1rem !important; }
        .input-section { padding: 1.5rem 1rem 1.5rem 1rem !important; }
        .hero-title { font-size: clamp(1.8rem, 8vw, 2.3rem) !important; margin-bottom: 1rem !important; }
        .hero-subtitle { font-size: 0.85rem !important; line-height: 1.6 !important; margin-bottom: 1.5rem !important; }
        .features { grid-template-columns: 1fr !important; gap: 1rem !important; padding: 1rem 0 !important; }
        .feature-card { padding: 1.25rem !important; }
        .answer-card { padding: 1.25rem !important; font-size: 0.85rem !important; line-height: 1.7 !important; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT — Two columns: hero left + interface right
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
        <div style="font-family: Inter, sans-serif; font-size: 0.85rem; color: #8896a7; margin-bottom: 1.5rem; line-height: 1.7;">
            Posez votre question ci-dessous. La réponse sera fondée uniquement sur les Écritures et les enseignements approuvés.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check admin URL
    is_admin = st.query_params.get("admin", "") == "true"

    if is_admin:
        tab1, tab2, tab3 = st.tabs(["Question", "Document", "Admin"])
    else:
        tab1, tab2 = st.tabs(["Question", "Document"])
        tab3 = None

    # ── TAB 1 — QUESTION ──
    with tab1:
        question = st.text_input(
            "Votre question",
            placeholder="Ex: Quel est le mystère de Dieu selon la Bible ?",
            key="input_question"
        )

        if st.button("Rechercher", key="search"):
            if question:
                with st.spinner("Recherche dans les Écritures..."):
                    answer = ai_search(question)
                answer_html = answer.replace("\n", "<br>")
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-label">✝ Réponse doctrinale</div>
                    <hr class="gold-divider">
                    {answer_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Veuillez entrer une question.")

    # ── TAB 2 — DOCUMENT ──
    with tab2:
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#8896a7;margin-bottom:1rem;'>Le document sera analysé et un rapport sera envoyé au responsable pour approbation.</p>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("Choisir un fichier PDF", type=["pdf"], key="pdf_uploader")
        submitter_email = st.text_input("Votre adresse email", placeholder="votre@email.com", key="input_email")

        if st.button("Soumettre pour validation", key="submit"):
            if uploaded_file and submitter_email:
                with st.spinner("Analyse doctrinale en cours..."):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # Récupération et nettoyage immédiat des astérisques du rapport
                    raw_report = validate_document(temp_path)
                    validation_report = raw_report.replace("**", "")

                    # Sauvegarde persistante dans la base JSON locale pour l'Admin
                    sauvegarder_document_attente(uploaded_file.name, submitter_email, validation_report, temp_path)

                send_notification_email(uploaded_file.name, validation_report)
                st.success("Document soumis. Le responsable a été notifié.")
                st.markdown("#### Rapport de validation")
                st.write(validation_report)
            else:
                st.warning("Veuillez télécharger un PDF et entrer votre email.")

    # ── TAB 3 — ADMIN ──
    if tab3 is not None:
        with tab3:
            if "admin_logged_in" not in st.session_state:
                st.session_state["admin_logged_in"] = False

            if not st.session_state["admin_logged_in"]:
                st.markdown(
                    "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#8896a7;'>Accès réservé au responsable de l'église.</p>",
                    unsafe_allow_html=True
                )
                if st.button("Envoyer un code de connexion", key="send_code"):
                    code = str(random.randint(100000, 999999))
                    st.session_state["login_code"] = code
                    sender = os.getenv("GMAIL_ADDRESS")
                    password = os.getenv("GMAIL_APP_PASSWORD")
                    receiver = os.getenv("FATHER_EMAIL")
                    msg = MIMEMultipart()
                    msg["From"] = sender
                    msg["To"] = receiver
                    msg["Subject"] = "Code de connexion — Assistant Doctrinal"
                    msg.attach(
                        MIMEText(f"Bonjour,\n\nVotre code : {code}\n\nCordialement,\nL'Assistant Doctrinal", "plain"))
                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                        server.starttls()
                        server.login(sender, password)
                        server.sendmail(sender, receiver, msg.as_string())
                    st.success("Code envoyé.")

                entered_code = st.text_input("Code reçu par email", placeholder="123456", key="admin_code_input")
                if st.button("Se connecter", key="login"):
                    if "login_code" in st.session_state and entered_code == st.session_state["login_code"]:
                        st.session_state["admin_logged_in"] = True
                        st.rerun()
                    else:
                        st.error("Code incorrect.")

            else:
                col_h, col_l = st.columns([4, 1])
                with col_h:
                    st.markdown("#### Documents en attente")
                with col_l:
                    if st.button("Déconnexion", key="logout"):
                        st.session_state["admin_logged_in"] = False
                        if "login_code" in st.session_state:
                            del st.session_state["login_code"]
                        st.rerun()

                # Lecture de la file d'attente partagée depuis le fichier JSON
                docs_a_valider = charger_documents_attente()

                if not docs_a_valider:
                    st.info("Aucun document en attente de validation.")
                else:
                    # Affichage propre en boucle sans astérisques Markdown
                    for index, doc in enumerate(docs_a_valider):
                        st.markdown(f"📄 <b>Document :</b> {doc['nom']}", unsafe_allow_html=True)
                        st.markdown(f"📧 <b>Soumis par :</b> {doc['email']}", unsafe_allow_html=True)
                        st.text_area("Rapport d'analyse", value=doc['rapport'], height=250, disabled=True,
                                     key=f"report_area_{index}")

                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button("✅ Approuver", key=f"approve_{index}"):
                                with st.spinner("Indexation du document..."):
                                    if os.path.exists(doc['temp_path']):
                                        chunks_added = add_to_collection(doc['temp_path'])
                                        os.remove(doc['temp_path'])
                                        st.success(f"Approuvé — {chunks_added} segments indexés dans la base globale.")
                                    else:
                                        st.error("Le fichier temporaire est introuvable sur le serveur.")
                                supprimer_document_attente(doc['nom'])
                                st.rerun()

                        with col2:
                            if st.button("❌ Rejeter", key=f"reject_{index}"):
                                sender = os.getenv("GMAIL_ADDRESS")
                                password_smtp = os.getenv("GMAIL_APP_PASSWORD")
                                receiver = doc["email"]
                                doc_name = doc["nom"]

                                msg = MIMEMultipart()
                                msg["From"] = sender
                                msg["To"] = receiver
                                msg["Subject"] = f"Document rejeté : {doc_name}"
                                msg.attach(MIMEText(
                                    f"Bonjour,\n\nVotre document « {doc_name} » n'a pas été approuvé par le comité doctrinal.\n\nCordialement,\nL'Assistant Doctrinal",
                                    "plain"
                                ))
                                try:
                                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                                        server.starttls()
                                        server.login(sender, password_smtp)
                                        server.sendmail(sender, receiver, msg.as_string())
                                except Exception as e:
                                    st.error(f"Erreur d'envoi d'email au soumetteur: {e}")

                                if os.path.exists(doc['temp_path']):
                                    os.remove(doc['temp_path'])

                                supprimer_document_attente(doc['nom'])
                                st.warning("Document rejeté. L'auteur a été notifié par email.")
                                st.rerun()

                        st.markdown("<hr style='border-top: 1px dashed #1a2540; margin: 2rem 0;'>",
                                    unsafe_allow_html=True)

# ============================================================
# BOTTOM BAR
# ============================================================
st.markdown("""
<div class="bottom-bar">
    Conseil : gardez toujours les références visibles pour encourager la vérification personnelle.
</div>
""", unsafe_allow_html=True)