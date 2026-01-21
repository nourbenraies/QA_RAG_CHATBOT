import sys
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.utils.embeddings import create_embeddings
from src.utils.vectorstore import create_retriever
from src.utils.file_handler import load_pdfs

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_Key")

st.set_page_config(page_title="QA_ Chatbot", layout="wide")
st.title("Bonjour ! Veuillez d’abord importer vos documents PDF pour commencer.")

# --- 1. Gestionnaires d'affichage (Callbacks) ---
# Ces classes permettent d'afficher le texte qui s'écrit petit à petit et les sources

class StreamHandler(BaseCallbackHandler):
    """Gère l'affichage du texte en streaming (effet machine à écrire)"""
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class PostMessageHandler(BaseCallbackHandler):
    """Gère l'affichage des sources (les morceaux de PDF utilisés) après la réponse"""
    def __init__(self, msg_container):
        self.msg_container = msg_container
        self.sources = []

    def on_retriever_end(self, documents, **kwargs):
        # Quand le retriever a fini, on récupère les documents trouvés
        for d in documents:
            metadata = {
                "source": os.path.basename(d.metadata.get("source", "Inconnu")),
                "page": d.metadata.get("page", 0),
                "content": d.page_content[:200] + "..." # Aperçu du texte
            }
            self.sources.append(metadata)

    def on_llm_end(self, response, **kwargs):
        # Quand le LLM a fini, on affiche un tableau avec les sources
        if self.sources:
            with self.msg_container:
                st.markdown("---")
                st.markdown("**🔍 Sources utilisées :**")
                st.dataframe(pd.DataFrame(self.sources[:3]), hide_index=True)

# --- 2. Initialisation des Ressources (Cache) ---

@st.cache_resource
def get_embedding_model():
    return create_embeddings()

embeddings_model = get_embedding_model()

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- 3. Sidebar : Chargement des fichiers ---
with st.sidebar:
    st.header(" Vos Documents")
    uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Traiter les documents"):
        with st.spinner("Analyse en cours..."):
            # A. Charger
            raw_docs = load_pdfs(uploaded_files)
            # B. Indexer (Vector Store)
            retriever = create_retriever(raw_docs, embeddings_model)
            # C. Sauvegarder
            st.session_state.retriever = retriever
            st.success("Documents indexés avec succès !")

# --- 4. Logique du Chatbot ---

# Si aucun document n'est chargé, on arrête l'affichage ici
if not st.session_state.retriever:
    st.stop()

# Initialisation du LLM (Groq)
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    streaming=True
)

# Template du Prompt
qa_template = """
Tu es un assistant expert. Utilise uniquement le contexte suivant pour répondre à la question.
Si tu ne connais pas la réponse d'après le contexte, dis simplement que tu ne sais pas.
Sois précis et concis.

Contexte :
{context}

Question : 
{question}
"""
prompt = ChatPromptTemplate.from_template(qa_template)

# Fonction pour formater les docs en string
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Création de la chaîne RAG
rag_chain = (
    {
        "context": st.session_state.retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# --- 5. Interface de Chat ---

# Gestion de l'historique des messages
msg_history = StreamlitChatMessageHistory(key="langchain_messages")

if len(msg_history.messages) == 0:
    msg_history.add_ai_message("Bonjour ! Posez-moi une question sur vos documents.")

# Affichage des anciens messages
for msg in msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

# Zone de saisie utilisateur
if user_question := st.chat_input("Votre question..."):
    
    # 1. Afficher la question de l'utilisateur
    st.chat_message("human").write(user_question)
    msg_history.add_user_message(user_question)

    # 2. Générer la réponse de l'IA
    with st.chat_message("ai"):
        # Conteneurs pour le stream et les sources
        response_placeholder = st.empty()
        sources_placeholder = st.container()
        
        # Initialisation des callbacks
        stream_handler = StreamHandler(response_placeholder)
        pm_handler = PostMessageHandler(sources_placeholder)
        
        # Lancement de la chaîne avec les callbacks
        response = rag_chain.invoke(
            user_question, 
            config={"callbacks": [stream_handler, pm_handler]}
        )
        
        # On sauvegarde la réponse complète dans l'historique
        msg_history.add_ai_message(response.content)