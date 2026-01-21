import os 
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs(uploaded_files):
    
    """ Charge les fichiers PDF uploadés et retourne une liste de documents"""
    docs = []

    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

            loader = PyMuPDFLoader(temp_filepath)
            docs.extend(loader.load())

    return docs






