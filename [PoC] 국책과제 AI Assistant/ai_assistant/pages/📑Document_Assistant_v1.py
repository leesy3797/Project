import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from langchain.memory import ConversationBufferMemory
from google import generativeai as genai
import os
from langsmith import Client
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from utils import *
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
import time
from streamlit_feedback import streamlit_feedback


# 환경 변수 로드
load_dotenv()

# API 키와 모델 정보
GEMINI_MODEL = 'gemini-1.5-flash-latest'
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
WELCOME_MESSAGE = "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"


# os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
logger.info(f"Langchain Project : {os.environ["LANGSMITH_PROJECT"]}")
logger.info(f"Langchain API : {os.environ["LANGSMITH_API_KEY"]}")

# Google AI 설정
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="Document Assistant V1",
    page_icon=":mag:",
    layout="centered",
    # layout='wide',
    initial_sidebar_state="auto",
    menu_items={
        'About': "# :green[무엇이든 물어보살]입니다.\n 문서 기반의 **AI 챗봇**으로 당신의 궁금증을 해소하세요!"
    }
)

st.title("🤖 도큐먼트 어시스턴트 (with Naive RAG) ✨")
st.subheader("Ask Anything to Me! 💡🔮")

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "langsmith_api_key" not in st.session_state:
    st.session_state.langsmith_api_key = os.environ["LANGSMITH_API_KEY"]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def check_if_key_exists(key):
    return key in st.session_state

if not check_if_key_exists("langsmith_api_key"):
    st.info(
        "⚠️ [LangSmith API Key](https://docs.smith.langchain.com/observability#2-create-an-api-key)를 추가해 주세요."
    )
else:
    # langchain_endpoint = st.secrets["LANGCHAIN_ENDPOINT"]
    langsmith_endpoint = os.environ["LANGSMITH_ENDPOINT"]

    client = Client(
        api_url = langsmith_endpoint, 
        api_key = st.session_state["langsmith_api_key"]
    )
    ls_tracer = LangChainTracer(
        # project_name=st.secrets["LANGCHAIN_PROJECT"],
        project_name=os.environ["LANGSMITH_PROJECT"],

        client=client
        )
    run_collector = RunCollectorCallbackHandler()
    cfg = RunnableConfig()
    cfg["callbacks"] = [ls_tracer, run_collector]

if not check_if_key_exists("google_api_key"):
    st.info(
       "⚠️ [Google API key](https://platform.openai.com/docs/guides/authentication) 를 추가해 주세요." 
    )  

with st.sidebar:
    st.header("설정")
    
    uploaded_files = st.file_uploader(
        "도큐먼트 파일 업로드",
        type=['pdf', 'docx', "pptx", "hwp", "hwpx"],
        accept_multiple_files=True,
    )
    
    google_api_key = None
    openai_api_key = None

    selected_model = st.selectbox(
        "챗봇 모델 선택",
        [None, "Gemini-1.5-Pro", "Gemini-1.5-Flash", "GPT-4o-mini", "Llama3"],
        format_func=lambda x: "모델을 선택하세요" if x is None else x
    )

    if (selected_model in ["Gemini-1.5-Pro", "Gemini-1.5-Flash"]) and (not google_api_key):
        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            value=GOOGLE_API_KEY
        )
        if "google_api_key" not in st.session_state:
            st.session_state["google_api_key"] = google_api_key

    elif (selected_model in ["GPT-4o-mini"]) and (not openai_api_key):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            key="openai_model_api_key",
            type="password",
            value = OPENAI_API_KEY
        )
        if "openai_api_key" not in st.session_state:
            st.session_state["openai_api_key"] = openai_api_key

    if google_api_key or openai_api_key or (selected_model == "Llama3"):

        selected_embedding_model = st.selectbox(
            "임베딩 모델 선택",
            [None, "GoogleEmbedding", "OpenAIEmbedding", "ko-sroberta-multitask"],
            format_func=lambda x: "임베딩을 선택하세요" if x is None else x
        )
        if selected_embedding_model:
            if selected_embedding_model == "GoogleEmbedding" and not google_api_key:
                google_api_key = st.text_input(
                    "Google API Key",
                    type="password",
                    value=GOOGLE_API_KEY
                )
                if "google_api_key" not in st.session_state:
                    st.session_state["google_api_key"] = google_api_key

            elif selected_embedding_model == "OpenAIEmbedding" and not openai_api_key:
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value = OPENAI_API_KEY
                )
                if "openai_api_key" not in st.session_state:
                    st.session_state["openai_api_key"] = openai_api_key

    process = st.button("설정 완료")
 

    reset_history = st.button("대화내용 초기화", type="primary")

    if reset_history:
        st.session_state.messages = []
        st.session_state["last_run"] = None

    api_keys = {
        "google_api_key": google_api_key,
        "openai_api_key": openai_api_key
    }

# Process 버튼을 누를 때 API 키 유효성 검증
if process:
    # st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": WELCOME_MESSAGE})
    
    check_api_keys(selected_model, selected_embedding_model, openai_api_key, google_api_key)

    with st.spinner("문서 처리 중..."):
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks, selected_embedding_model, api_keys)
        st.session_state.conversation, st.session_state.retriever = create_rag_chain(text_chunks, vectorstore, selected_model, api_keys)
        st.session_state.processComplete = True 

    with st.chat_message("assistant"):
        typing_area = st.empty()  # 빈 공간을 만들어서 텍스트를 업데이트할 준비
        typing_text = ""
        for char in WELCOME_MESSAGE:
            typing_text += char
            typing_area.markdown(typing_text)  # 현재까지의 텍스트를 출력
            time.sleep(0.02)  # 각 글자를 출력할 때마다 약간의 지연 (0.05초)
        st.rerun()
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat logic
if query := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        chain = st.session_state.conversation
        retriever = st.session_state.retriever

        with st.spinner("생각중..."):
            memory = st.session_state.memory
            retrieved_docs = retriever.invoke(query)
            response = chain.invoke(query, config=cfg)
            memory.save_context(
                inputs = {"user" : query},
                outputs = {"assistant" : response}
            )

        # 타이핑 효과로 출력
        typing_area = st.empty()  # 빈 공간을 만들어서 텍스트를 업데이트할 준비
        typing_text = ""
        for char in response:
            typing_text += char
            typing_area.markdown(typing_text)  # 현재까지의 텍스트를 출력
            time.sleep(0.02)  # 각 글자를 출력할 때마다 약간의 지연 (0.05초)

        with st.expander(":mag:**출처 확인**"):
            for doc in retrieved_docs:
                st.markdown(f"{doc.metadata['source'].split("/")[-1]}의 {doc.metadata["page"] + 1}페이지", help=doc.page_content)
            
        st.session_state.messages.append({"role": "assistant", "content": response})

    wait_for_all_tracers()
    logger.info(f"## 트레이스 : {run_collector.traced_runs}")
    st.session_state.last_run = run_collector.traced_runs[0].id

@st.cache_data(ttl="2h", show_spinner=False)
def get_run_url(run_id):
    time.sleep(1)
    return client.read_run(run_id).url

if st.session_state.get("last_run"):
    run_url = get_run_url(st.session_state.last_run)
    st.sidebar.markdown(f"[LangSmith 추적🛠️]({run_url})")
    feedback = streamlit_feedback(
        feedback_type = "thumbs",
        optional_text_label = None,
        key = f"feedback_{st.session_state.last_run}"
    )
    if feedback:
        scores = {"👍": 1, "👎": 0}
        client.create_feedback(
            st.session_state.last_run,
            feedback["type"],
            score=scores[feedback["score"]],
            comment=query
        )
        st.toast("피드백을 저장하였습니다.!", icon="📝")