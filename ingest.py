import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from openai import OpenAI

# === Charger la clé OpenAI ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Étape 1 : Chargement des documents ===
def load_documents():
    print("➡️ Chargement des documents…")
    docs = []

    files_to_load = [
        "data/yasmine_cv.pdf",
        "data/bio.txt",
        "data/competencies_folder.txt"
    ]

    for file_path in files_to_load:
        if not os.path.exists(file_path):
            print(f"⚠️ Fichier introuvable : {file_path}")
            continue

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            print(f"⚠️ Format non supporté : {file_path}")
            continue

        file_docs = loader.load()
        docs.extend(file_docs)
        print(f"   → {os.path.basename(file_path)} : {len(file_docs)} sections")

    print(f"➡️ Total documents chargés : {len(docs)}")
    return docs

# === Étape 2 : Découpage des textes ===
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"➡️ Découpage terminé : {len(chunks)} chunks")
    return chunks

# === Étape 3 : Nouvelle classe d'embeddings compatible OpenAI >=1.30 ===
class MyEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts):
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text):
        return self._embed_text(text)

    def _embed_text(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",  # modèle le plus récent et économique
            input=text
        )
        return response.data[0].embedding

# === Étape 4 : Création de l’index FAISS ===
def create_faiss_index(chunks):
    print("➡️ Création de l’index FAISS…")
    embeddings = MyEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ Index sauvegardé dans ./faiss_index")

# === Main ===
def main():
    docs = load_documents()
    if not docs:
        print("❌ Aucun document valide trouvé. Vérifie le dossier data/.")
        return
    chunks = split_documents(docs)
    create_faiss_index(chunks)

if __name__ == "__main__":
    main()
