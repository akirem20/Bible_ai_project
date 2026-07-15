import streamlit as st
import os
from main import ai_search, validate_document, add_to_collection, send_notification_email
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Assistant Doctrinal",
    page_icon="✝️",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;500&display=swap');

    .stApp { background-color: #080d1a; color: #f0ece0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

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
</style>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT — Two columns: hero left + interface right
# ============================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="padding: 4rem 2rem 2rem 2rem;">
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
    <div style="padding: 4rem 2rem 2rem 2rem;">
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
            placeholder="Ex: Quel est le mystère de Dieu selon la Bible ?"
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
        uploaded_file = st.file_uploader("Choisir un fichier PDF", type=["pdf"])
        submitter_email = st.text_input("Votre adresse email", placeholder="votre@email.com")

        if st.button("Soumettre pour validation", key="submit"):
            if uploaded_file and submitter_email:
                with st.spinner("Analyse doctrinale en cours..."):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                    validation_report = validate_document(temp_path)
                    st.session_state["pending_doc"] = temp_path
                    st.session_state["pending_report"] = validation_report
                    st.session_state["pending_name"] = uploaded_file.name
                    st.session_state["submitter_email"] = submitter_email
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
                    msg.attach(MIMEText(f"Bonjour,\n\nVotre code : {code}\n\nCordialement,\nL'Assistant Doctrinal", "plain"))
                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                        server.starttls()
                        server.login(sender, password)
                        server.sendmail(sender, receiver, msg.as_string())
                    st.success("Code envoyé.")

                entered_code = st.text_input("Code reçu par email", placeholder="123456")
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

                if "pending_doc" not in st.session_state:
                    st.info("Aucun document en attente de validation.")
                else:
                    st.markdown(f"**Document :** {st.session_state['pending_name']}")
                    st.write(st.session_state["pending_report"])
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Approuver", key="approve"):
                            with st.spinner("Indexation..."):
                                chunks_added = add_to_collection(st.session_state["pending_doc"])
                            if os.path.exists(st.session_state["pending_doc"]):
                                os.remove(st.session_state["pending_doc"])
                            for key in ["pending_doc", "pending_report", "pending_name"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.success(f"Approuvé — {chunks_added} chunks indexés.")
                    with col2:
                        if st.button("❌ Rejeter", key="reject"):
                            if "submitter_email" in st.session_state:
                                sender = os.getenv("GMAIL_ADDRESS")
                                password_smtp = os.getenv("GMAIL_APP_PASSWORD")
                                receiver = st.session_state["submitter_email"]
                                doc_name = st.session_state["pending_name"]
                                msg = MIMEMultipart()
                                msg["From"] = sender
                                msg["To"] = receiver
                                msg["Subject"] = f"Document rejeté : {doc_name}"
                                msg.attach(MIMEText(f"Bonjour,\n\nVotre document « {doc_name} » n'a pas été approuvé.\n\nCordialement,\nL'Assistant Doctrinal", "plain"))
                                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                                    server.starttls()
                                    server.login(sender, password_smtp)
                                    server.sendmail(sender, receiver, msg.as_string())
                            if os.path.exists(st.session_state["pending_doc"]):
                                os.remove(st.session_state["pending_doc"])
                            for key in ["pending_doc", "pending_report", "pending_name", "submitter_email"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.warning("Document rejeté. L'auteur a été notifié.")

# ============================================================
# BOTTOM BAR
# ============================================================
st.markdown("""
<div class="bottom-bar">
    Conseil : gardez toujours les références visibles pour encourager la vérification personnelle.
</div>
""", unsafe_allow_html=True)