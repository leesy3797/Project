import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st
import tiktoken
from loguru import logger

import time
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import os

# API 키 검증 함수
def check_api_keys(selected_model, selected_embedding_model, openai_api_key, google_api_key):
    if selected_model == "GPT-4o" or selected_embedding_model == "OpenAIEmbedding":
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

    if selected_model in ["Gemini-1.5-Pro", "Gemini-1.5-Flash"] or selected_embedding_model == "GoogleEmbedding":
        if not google_api_key:
            st.info("Please add your Google API key to continue.")
            st.stop()

# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.02)

# 텍스트 요약 함수
def summarize_text(text, llm):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(text)
    return summary


def get_hybrid_retriever(text_chunks, vectordb, k=5, bm25_r=0.5):
    bm25_retriever = BM25Retriever.from_documents(
        text_chunks
        )
    bm25_retriever.k = 5
    faiss_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20
        }
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[bm25_r, 1 - bm25_r]
    )
    return ensemble_retriever


def get_vectorstore(text_chunks, embedding_model_name, api_keys):
    if embedding_model_name == "GoogleEmbedding":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_keys["google_api_key"]
        )
    elif embedding_model_name == "OpenAIEmbedding":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_keys["openai_api_key"]
        )
    else:  # "ko-sroberta-multitask"
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    return FAISS.from_documents(text_chunks, embeddings)

# RAG 체인 생성 함수
def create_rag_chain(text_chunks, vectorstore, selected_model, api_keys):
    logger.info(f"Google API Key : {api_keys["google_api_key"]}")
    logger.info(f"OpenAI API Key : {api_keys["openai_api_key"]}")

    # LLM 선택 로직
    llm_model = {
        "Gemini-1.5-Pro": "gemini-1.5-pro-latest",
        "Gemini-1.5-Flash": "gemini-1.5-flash-latest",
        "GPT-4o-mini": "gpt-4o-mini",
        "Llama3": "llama3:instruct"
    }

    if selected_model in ["Gemini-1.5-Pro", "Gemini-1.5-Flash"]:
        llm = ChatGoogleGenerativeAI(
            model=llm_model[selected_model],
            api_key=api_keys["google_api_key"]
        )
    elif selected_model == "GPT-4o-mini":
        llm = ChatOpenAI(
            model_name=llm_model[selected_model],
            openai_api_key=api_keys["openai_api_key"]
        )
    else:  # "llama3"
        llm = ChatOllama(
            model=llm_model[selected_model]
        )

    # 대화 메모리 설정
    memory = st.session_state.memory
    
    # RAG 프롬프트 설정
    prompt_template = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=(
            "You are having a conversation with a user. Use the following chat history and context to answer the question.\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    get_chat_history = RunnableLambda(lambda inputs: memory.chat_memory)

    # RAG 체인 구성
    retriever = get_hybrid_retriever(text_chunks, vectorstore)
    rag_chain = {
        "context": retriever,
        "question": RunnablePassthrough(),
        "chat_history": get_chat_history
    } | prompt_template | llm | StrOutputParser()

    return rag_chain, retriever

def get_text(docs):
    doc_list = []
    directory_path = "./data/"

    # 디렉토리 없을 시 디렉토리 추가가
    os.makedirs(directory_path, exist_ok=True)

    for doc in docs:
        file_name = os.path.join(directory_path, doc.name)
        logger.info(f"저장 경로 : {file_name}")
        with open(file_name, 'wb') as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token:str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def print_messages():
    if ("messages" in st.session_state) and (len(st.session_state["messages"]) > 0):
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)