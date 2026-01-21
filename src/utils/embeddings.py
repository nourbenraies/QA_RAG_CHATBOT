from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings():
    """ initialisation des embeddings à partir d'un modèle HuggingFace """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )