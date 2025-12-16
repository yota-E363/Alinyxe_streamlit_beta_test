import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
import docx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

from openai import OpenAI
import google.generativeai as genai

# =========================
# CONFIGURATION GÉNÉRALE
# =========================
st.set_page_config(
    page_title="Analyse intelligente de documents",
    page_icon="📊",
    layout="wide"
)

load_dotenv()

# =========================
# FONCTIONS EXTRACTION TEXTE
# =========================
def extract_pdf(file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        path = tmp.name

    pdf = fitz.open(path)
    for i, page in enumerate(pdf, start=1):
        text += f"\n\n=== [PAGE {i}] ===\n" + page.get_text()

    os.unlink(path)
    return text, len(pdf)

def extract_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs), 1

def extract_txt(file):
    return file.read().decode("utf-8"), 1

# =========================
# IA UNIFIÉE
# =========================
def llm_response(prompt, provider, api_key):
    instructions = (
        "Tu es analyste professionnel. "
        "N'invente aucune donnée. "
        "Si une info manque : 'non précisé'."
    )

    if provider == "OpenAI":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content

    if provider == "Gemini":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(instructions + "\n\n" + prompt).text

# =========================
# SIDEBAR CONFIGURATION
# =========================
with st.sidebar:
    st.header("⚙️ Configuration IA")

    provider = st.selectbox("Fournisseur IA", ["OpenAI", "Gemini"])

    api_key = st.text_input(
        f"Clé API {provider}",
        type="password",
        value=os.getenv(f"{provider.upper()}_API_KEY", "")
    )

    if not api_key:
        st.warning("Veuillez fournir une clé API")

# =========================
# TABS PRINCIPAUX
# =========================
tab1, tab2, tab3 = st.tabs([
    "📂 Upload fichiers",
    "📊 Dashboard",
    "💬 Analyse & Questions"
])

# =========================
# TAB 1 — UPLOAD
# =========================
with tab1:
    st.header("📂 Upload de documents")

    uploaded_files = st.file_uploader(
        "Téléversez vos fichiers (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    full_text = ""
    stats = []

    if uploaded_files:
        for file in uploaded_files:
            if file.type == "application/pdf":
                text, pages = extract_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text, pages = extract_docx(file)
            else:
                text, pages = extract_txt(file)

            stats.append({
                "Fichier": file.name,
                "Pages": pages,
                "Caractères": len(text)
            })
            full_text += "\n" + text

        st.success(f"{len(uploaded_files)} fichiers chargés")
        st.session_state["text"] = full_text
        st.session_state["stats"] = pd.DataFrame(stats)

# =========================
# TAB 2 — DASHBOARD
# =========================
with tab2:
    st.header("📊 Dashboard")

    if "stats" in st.session_state:
        df = st.session_state["stats"]

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Fichiers", len(df))
        col2.metric("📚 Pages totales", df["Pages"].sum())
        col3.metric("✍️ Caractères", df["Caractères"].sum())

        fig, ax = plt.subplots()
        df.plot(kind="bar", x="Fichier", y="Caractères", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Chargez des fichiers pour voir le dashboard")

# =========================
# TAB 3 — ANALYSE IA
# =========================
with tab3:
    st.header("💬 Analyse intelligente")

    if "text" not in st.session_state:
        st.info("Veuillez uploader des fichiers")
    else:
        if st.button("🧠 Générer un résumé"):
            with st.spinner("Analyse en cours..."):
                summary = llm_response(
                    "Fais un résumé structuré du document :\n" + st.session_state["text"],
                    provider,
                    api_key
                )
            st.subheader("📄 Résumé")
            st.markdown(summary)
            st.download_button(
                "Télécharger le résumé",
                summary,
                file_name="resume.md"
            )

        st.markdown("---")
        question = st.text_input("❓ Poser une question")

        if question:
            with st.spinner("Recherche..."):
                answer = llm_response(
                    f"QUESTION : {question}\n\nDOCUMENT :\n{st.session_state['text']}",
                    provider,
                    api_key
                )
            st.markdown("### Réponse")
            st.markdown(answer)
