
import os
import streamlit as st

# Streamlit Part
st.title('Chat with Quran')

# Initialize a list to store the history of prompts and responses
if 'history' not in st.session_state:
    st.session_state['history'] = []

prompt = st.text_input("Say something")
response = None
if prompt:
    # Add the user's prompt to the history
    st.session_state['history'].append(("user", prompt))
    response = f'RESPONSE {prompt} RESPONSE'
    # Add the AI's response to the history
    st.session_state['history'].append(("ai", response))

# Display the history of prompts and responses
for speaker, text in st.session_state['history']:
    with st.chat_message(speaker, avatar=None):
        st.write(text)
