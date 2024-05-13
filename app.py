import os
import streamlit as st
from dotenv import load_dotenv

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings


load_dotenv()

# Intializing OpenAI LLM
llm = OpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'))

# Intializing OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))

# Intializaing Chroma Vector Store
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever=vector_store.as_retriever()
chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Streamlit Part
st.title('Chat with Quran')

if 'history' not in st.session_state:
    st.session_state['history'] = []

prompt = st.chat_input("Ask Something")
response = None
if prompt:
    st.session_state['history'].append(("user", prompt))
    response = chain.run(prompt)
    st.session_state['history'].append(("ai", response))



for speaker, text in st.session_state['history']:
    with st.chat_message(speaker, avatar=None):
        st.write(text)
