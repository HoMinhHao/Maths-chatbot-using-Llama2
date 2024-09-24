from src.helper import *
from langchain_community.vectorstores import Pinecone as PC
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

extracted_data = load_pdf("data/")
text_chunks=text_split(extracted_data)
embedding=download_huggingface_embedding()

index_name="maths-chatbot"
docsearch=PC.from_texts([t.page_content for t in text_chunks], embedding, index_name=index_name)
