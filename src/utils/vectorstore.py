import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma



def create_retriever(documents,embeddings_model , collection_name="document_collection", db_path="./chroma_db"):

    """ Crée un retriever à partir des documents et du modèle d'embeddings fourni """

    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(documents)



    vectordb = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings_model,  
        collection_name="document_collection",
        persist_directory=db_path
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})




     