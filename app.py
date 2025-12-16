import streamlit as st
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import tempfile
import pathlib

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Documents Financiers",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("📊 Analyse Automatique de Documents Financiers")
st.markdown("Transformez vos rapports financiers en résumés structurés grâce à l'IA générative")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")


    # Chargement des variables d'environnement
    # Chercher et charger le .env (remonte les dossiers si besoin)
    from dotenv import find_dotenv
    env_path = find_dotenv(filename=".env", usecwd=True)
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Interface pour configurer la clé API
    st.subheader("🔑 Configuration API OpenAI")
    
    # Charger la clé API existante depuis .env ou session
    default_api_key = os.getenv("OPENAI_API_KEY", "")
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = default_api_key
    
    # Champ de saisie pour la clé API
    api_key = st.text_input(
        "Clé API OpenAI",
        value=st.session_state.openai_api_key,
        type="password",
        placeholder="sk-...",
        help="Entrez votre clé API OpenAI. Elle sera sauvegardée pour cette session."
    )
    
    # Mettre à jour la session
    if api_key != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key
        st.success("✅ Clé API mise à jour !")
    
    # Vérification de la clé
    if not api_key:
        st.error("❌ Veuillez entrer votre clé API OpenAI")
        st.info("Vous pouvez obtenir une clé sur : https://platform.openai.com/api-keys")
        st.stop()
    else:
        st.success(f"✅ API Key configurée: {api_key[:8]}...")
    
    # Sélection du modèle
    model = st.selectbox(
        "Modèle OpenAI",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    
    # Limite de longueur du texte
    max_length = st.slider(
        "Longueur maximale du texte (caractères)",
        min_value=50000,
        max_value=200000,
        value=120000,
        step=10000
    )
    
    st.markdown("---")
    st.markdown("**Instructions :**")
    st.markdown("1. Uploadez votre PDF financier")
    st.markdown("2. Obtenez un résumé structuré")
    st.markdown("3. Posez des questions spécifiques")

# Fonction pour extraire le texte du PDF
def extract_pdf_text(pdf_file, max_length=120000):
    """Extrait le texte d'un PDF avec repères de pages"""
    try:
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        # Ouvrir le PDF avec PyMuPDF
        pdf = fitz.open(tmp_path)
        
        # Extraire le texte page par page
        text = ""
        for i, page in enumerate(pdf, start=1):
            page_text = page.get_text()
            text += f"\n\n=== [PAGE {i}] ===\n" + page_text.strip()
        
        # Nettoyer le texte
        text = "\n".join(line.strip() for line in text.splitlines())
        
        # Limiter la longueur si nécessaire
        if len(text) > max_length:
            text = text[:max_length]
            st.warning(f"⚠️ Le texte a été tronqué à {max_length} caractères pour éviter les dépassements d'API")
        
        # Nettoyer le fichier temporaire
        os.unlink(tmp_path)
        
        return text, len(text)
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du PDF: {str(e)}")
        return None, 0

# Fonction pour générer le résumé
def generate_summary(text, model="gpt-4o-mini"):
    """Génère un résumé financier structuré"""
    
    # Récupérer la clé API depuis la session
    api_key = st.session_state.get('openai_api_key')
    if not api_key:
        st.error("❌ Clé API non configurée")
        return None
    
    # Consignes pour le modèle
    instructions = (
        "Tu es analyste financier. On te fournit le texte d'un document financier "
        "(rapport annuel, trimestriel, comptes, bilan, annexes).\n\n"
        "Produis une synthèse **précise et chiffrée** en Markdown selon ce cadre :\n\n"
        "- **Société / Période / Devise** : (si repérable)\n"
        "- **Résumé exécutif (5–8 lignes)** : activité, faits marquants, contexte\n"
        "- **Chiffres clés** (tableau) :\n"
        " | Indicateur | Valeur | Évolution/Contexte | Période | Page |\n"
        " |---|---:|---|---|---:|\n"
        " (exemples : Chiffre d'affaires, EBIT/EBITDA, Résultat net, Marge, FCF, CAPEX, "
        "Dette nette, Trésorerie, NPL/Coût du risque pour banque, CET1, LCR/NSFR, etc.)\n"
        "- **Analyse** :\n"
        " - Performance (croissance, marges, cash)\n"
        " - Structure financière (dette, liquidité)\n"
        " - Risques & incertitudes (marché, réglementation, change)\n"
        " - Outlook / Guidance (si communiqué)\n"
        "- **Références internes** : pages/sections à relire\n\n"
        "Exigences :\n"
        "- **N'invente aucun chiffre**. Si une valeur n'apparaît pas clairement : `non précisé`.\n"
        "- Cite la **Page** d'origine quand c'est possible (repère `=== [PAGE X] ===`).\n"
        "- 6 à 12 **indicateurs quantitatifs** maximum (les plus utiles).\n"
        "- Reste concis : 200–350 mots hors tableau."
    )
    
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération du résumé: {str(e)}")
        return None

# Fonction pour répondre aux questions
def answer_question(text, question, model="gpt-4o"):
    """Répond à une question spécifique sur le contenu du PDF"""
    
    # Récupérer la clé API depuis la session
    api_key = st.session_state.get('openai_api_key')
    if not api_key:
        st.error("❌ Clé API non configurée")
        return None
    
    instructions = (
        "Tu es analyste financier. On te donne un extrait de rapport financier. "
        "Réponds uniquement à la question posée, sans inventer de données. "
        "Si la réponse n'est pas claire dans le texte, écris : 'non précisé'. "
        "Quand c'est possible, indique aussi la page d'origine (repère '=== [PAGE X] ===')."
    )
    
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"Question : {question}\n\nTexte PDF :\n{text}"}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la réponse à la question: {str(e)}")
        return None

# Interface principale
def main():
    # Onglets pour organiser l'interface
    tab1, tab2 = st.tabs(["📄 Upload & Analyse", "❓ Questions"])
    
    with tab1:
        st.header("📄 Upload et Analyse du PDF")
        
        # Upload du fichier
        uploaded_file = st.file_uploader(
            "Choisissez votre document financier (PDF)",
            type=['pdf'],
            help="Formats acceptés : PDF uniquement"
        )
        
        if uploaded_file is not None:
            # Informations sur le fichier
            file_details = {
                "Nom du fichier": uploaded_file.name,
                "Taille": f"{uploaded_file.size / 1024:.1f} KB",
                "Type": uploaded_file.type
            }
            st.json(file_details)
            
            # Bouton pour analyser
            if st.button("🚀 Analyser le document", type="primary"):
                with st.spinner("📖 Extraction du texte en cours..."):
                    text, text_length = extract_pdf_text(uploaded_file, max_length)
                
                if text:
                    st.success(f"✅ Texte extrait : {text_length} caractères")
                    
                    # Aperçu du texte
                    with st.expander("👁️ Aperçu du texte extrait"):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)
                    
                    # Génération du résumé
                    with st.spinner("🤖 Génération du résumé en cours..."):
                        summary = generate_summary(text, model)
                    
                    if summary:
                        st.success("✅ Résumé généré avec succès !")
                        
                        # Affichage du résumé
                        st.subheader("📊 Résumé Financier")
                        st.markdown(summary)
                        
                        # Stockage en session pour les questions
                        st.session_state['pdf_text'] = text
                        st.session_state['summary'] = summary
                        
                        # Téléchargement du résumé
                        st.download_button(
                            label="💾 Télécharger le résumé (Markdown)",
                            data=summary,
                            file_name=f"resume_{uploaded_file.name.replace('.pdf', '')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error("❌ Échec de la génération du résumé")
                else:
                    st.error("❌ Échec de l'extraction du texte")
    
    with tab2:
        st.header("❓ Questions sur le Document")
        
        if 'pdf_text' not in st.session_state:
            st.info("ℹ️ Veuillez d'abord analyser un document dans l'onglet 'Upload & Analyse'")
        else:
            st.success("✅ Document chargé et prêt pour les questions")
            
            # Interface de questions
            question = st.text_input(
                "Posez votre question sur le document :",
                placeholder="Ex: Quel est le chiffre d'affaires ? Quelle est la marge nette ?"
            )
            
            if question:
                if st.button("🔍 Rechercher la réponse", type="primary"):
                    with st.spinner("🤖 Recherche en cours..."):
                        answer = answer_question(st.session_state['pdf_text'], question, model)
                    
                    if answer:
                        st.success("✅ Réponse trouvée !")
                        st.markdown("**Question :** " + question)
                        st.markdown("**Réponse :**")
                        st.markdown(answer)
                    else:
                        st.error("❌ Échec de la recherche de réponse")
            
            # Questions suggérées
            st.subheader("💡 Questions suggérées")
            suggested_questions = [
                "Quel est le chiffre d'affaires ?",
                "Quelle est la marge nette ?",
                "Quels sont les principaux risques identifiés ?",
                "Quelle est la dette nette ?",
                "Quel est le cash flow opérationnel ?"
            ]
            
            for i, suggested_q in enumerate(suggested_questions):
                if st.button(f"❓ {suggested_q}", key=f"suggested_{i}"):
                    with st.spinner("🤖 Recherche en cours..."):
                        answer = answer_question(st.session_state['pdf_text'], suggested_q, model)
                    
                    if answer:
                        st.success("✅ Réponse trouvée !")
                        st.markdown("**Question :** " + suggested_q)
                        st.markdown("**Réponse :**")
                        st.markdown(answer)
                    else:
                        st.error("❌ Échec de la recherche de réponse")

# Footer
st.markdown("---")
st.markdown(
    "**Note importante :** Vérifiez toujours les chiffres affichés et leurs pages d'origine. "
    "En cas d'ambiguïté dans le PDF, utilisez 'non précisé' et confirmez dans le document source."
)

if __name__ == "__main__":
    main()
