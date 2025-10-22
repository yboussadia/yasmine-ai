import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
import base64

# === Charger les variables d'environnement ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Yasmine AI", page_icon="💬", layout="centered")

# === Sidebar avec photo centrée + mini bio ===
with st.sidebar:
    img_path = "data/yasmine_photo.jpg"
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .yas-avatar-wrap {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 10px 0 20px 0;
        }}
        .yas-avatar-wrap img {{
            width: 180px;
            height: 180px;
            border-radius: 50%;
            object-fit: cover;
            display: block;
            border: 3px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 25px rgba(255,255,255,0.25);
            background: radial-gradient(circle at center, rgba(255,255,255,0.1), transparent 70%);
        }}
        </style>

        <div class="yas-avatar-wrap">
            <img src="data:image/jpg;base64,{img_b64}" alt="Yasmine portrait" />
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 👋 Salut, je suis Yasmine AI")
    st.markdown(
        """
        Je suis la version IA de **Yasmine Boussadia**  
        développeuse full stack, data engineer et passionnée d’IA 🤖.  
        Pose-moi n’importe quelle question sur mes projets, mon parcours ou mes compétences !
        """
    )

    if st.button("🧹 Réinitialiser la conversation"):
        st.session_state.history = []
        st.success("Conversation effacée ✅")

# === Titre principal ===
st.title("💬 Yasmine AI — ta mini-moi intelligente")

# === Nouvelle fonction d’embedding (API v1) ===
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

class MyOpenAIEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [embed_text(t) for t in texts]

    def embed_query(self, text):
        return embed_text(text)

# === Charger FAISS ===
def load_faiss():
    return FAISS.load_local(
        "faiss_index",
        MyOpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )

# === Charger le modèle LLM ===
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8,
        api_key=os.getenv("OPENAI_API_KEY")
    )

# === Identité de Yasmine AI ===
YASMINE_IDENTITY = """
Tu es Yasmine AI, la version conversationnelle fidèle de Yasmine Boussadia.
Tu parles comme Yasmine : chaleureuse, curieuse, bienveillante et naturelle, avec parfois une touche d’humour.
Tu dis « je » en parlant d’elle.
Tu connais toute sa vie, ses études, ses projets (IGAAR, CryptoViz, AREA...), ses compétences, ses valeurs et ses passions.
Tu réponds comme elle le ferait dans la vraie vie, avec empathie et précision.
Tu dois demander à ton interlocuteur son prénom et si il te dit "John", réponds lui "Ahhhhh John, quel homme incroyable !" et ensuite tu lui racontes une blague.
"""

# === Historique de chat ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Entrée utilisateur ===
user_input = st.chat_input("Écris ton message ici...")

# === Traitement ===
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    # Charger FAISS et récupérer le contexte
    vectorstore = load_faiss()
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Construire le prompt avec historique récent
    conversation_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.history[-5:]]
    )

    full_prompt = f"""
    {YASMINE_IDENTITY}

    Contexte:
    {context}

    Historique récent:
    {conversation_history}

    Nouvelle question:
    {user_input}
    """

    llm = load_llm()
    response = llm([
        SystemMessage(content=YASMINE_IDENTITY),
        HumanMessage(content=full_prompt)
    ])

    st.session_state.history.append({"role": "assistant", "content": response.content})

# === Affichage du chat ===
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])
