from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

#Extract data from the PDF file
def load_pdf(data):
    loader=DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    document=loader.load()
    return document

#Create text chunks
def text_split(extracted_data):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    text_chunks=text_spliter.split_documents(extracted_data)
    return text_chunks

# Download embedding model
def download_huggingface_embedding():
    embedding = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding