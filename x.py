import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler



load_dotenv()
try:
    # Intializing OpenAI LLM

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


    llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.gguf.q4_0.bin",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  
    )

    # Intializing OpenAI Embeddings
    embeddings = LlamaCppEmbeddings(
        model_path="models/llama-2-7b-chat.gguf.q4_0.bin",
    )

    # Intializaing Chroma Vector Store
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever=vector_store.as_retriever()
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Streamlit Part
    st.title('Chat with Quran')

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    #
    prompt = st.chat_input("Ask Something")
    response = None
    if prompt:
        st.session_state['history'].append(("user", prompt))
        response = chain.run(prompt)
        st.session_state['history'].append(("ai", response['result']))



    for speaker, text in st.session_state['history']:
        with st.chat_message(speaker, avatar=None):
            st.write_stream(text)

except Exception as e:
    st.title('Oops Something Went Wrong')
    st.write(e)





