# Bring in deps
import os 
from bardapi import Bard
import streamlit as st
from streamlit_chat import message
from langchain.prompts import PromptTemplate
import requests


os.environ['_BARD_API_KEY']= 'cAjdtPrjFPpmCRMAnHLZBD3wa5d91yOt4oITLNIIqSEuYWvAAvljnV0z19Op50TPMzLVDA.'


st.title('personal nutrition doc')
title_template = PromptTemplate(
    input_variables = ['nutrient'],
    template = 'is {nutrient} crucial for my health?'
)

def response_api(answer):
    message = Bard().get_answer(str(answer))['content']
    return message

def user_input():
    input_text =  st.text_input('attach your nutrition facts ') 
    return input_text

if 'generate' not in st.session_state:
    st.session_state['generate']=[]
if 'past' not in st.session_state:
    st.session_state['past']=[]

user_text = user_input()

if user_text:
    output = response_api(user_text)
    st.session_state.generate.append(output)
    st.session_state.past.append(user_text)

    print("User Input:", user_text)
    print("Chatbot Output:", output)


# chat history
if st.session_state['generate']:

    for i in range(len(st.session_state['generate'])-1,-1,-1):
        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generate"][i],key=str(i) )

