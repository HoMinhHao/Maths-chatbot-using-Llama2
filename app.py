from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embedding
from langchain.vectorstores import Pinecone as PC
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.llms import CTransformers
from src.prompt import *
from langchain.chains import RetrievalQA
import os

app=Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

embedding=download_huggingface_embedding()

index_name="maths-chatbot"
docsearch=PC.from_existing_index(index_name, embedding)

prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
chain_type_kwargs={"prompt":prompt}

llm=CTransformers(model="model/llama-2-7b.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':600,
                          'temperature':0.5})

qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)