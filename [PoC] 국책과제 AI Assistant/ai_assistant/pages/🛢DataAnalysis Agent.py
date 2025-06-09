import streamlit as st
import pandas as pd
import logging
from dotenv import load_dotenv
import os
from google import generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import numpy as np
from langchain.prompts import PromptTemplate
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import create_react_agent
from langchain.chains import create_sql_query_chain
from ast import literal_eval
from loguru import logger
from langchain_community.chat_models import ChatOllama
from eralchemy import render_er
from langchain.callbacks.base import BaseCallbackHandler
import time

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash-latest"
)

# llm = ChatOllama(
#     model = "llama3:instruct"
# )

st.set_page_config(
    page_title = "DataAnalysis Agent",
    page_icon = "🛢"
    )
st.header("DataAnalysis Agent with Database(SQL)")
st.write('Enable the chatbot to interact with a SQL database through simple, conversational commands.')

dataframe = None
radio_opt = ["샘플 DB 사용 (Chinook DB)", "DB 직접 연결", "파일 업로드 (DB 변환)"]

tab1, tab2, tab3 = st.tabs(["데이터", "검색", "인사이트"])
db_uri = None

if 'uploaded_file_name' not in st.session_state:
    st.session_state["uploaded_file_name"] = "No DB or Files are linked"
if 'selected_opt' not in st.session_state:
    st.session_state["selected_opt"] = -1

class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = None

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can 
         record it as the final sql"""

        if action.tool == "sql_db_query":
            self.sql_result = action.tool_input

template = '''For the given question, first create a syntactically 
correct SQL query based on the provided {table_info} and its structure. 
Then, look at the results of the query and provide a concise answer based on the output. 
The answer must always be in bullet points and in Korean (Hangul).

The format should be as follows:

Question: "Insert the question here"
SQLQuery: "SQL query to run"
SQLResult: "Result of the SQL query"
Answer: "Final answer"

You can only use the following tables:

{table_info}

Question: {input}
'''
prompt = PromptTemplate.from_template(template)

with st.sidebar:
    selected_opt = st.radio(
        label = "사용하고자 하는 옵션 선택",
        options=radio_opt,
    )

    logger.info(f"라디오 선택 : {selected_opt}, 타입 : {type(selected_opt)}")

    if radio_opt.index(selected_opt) == 0:
        db_file = os.path.abspath("./sample/chinook.db")
        db_uri = f"sqlite:///{db_file.split("/")[-1]}"
        logger.info(f"파일 위치 : {db_file}")
        logger.info(f"DB URI : {db_uri}")
        sample_db_file = db_file.split("\\")[-1]

    elif radio_opt.index(selected_opt) == 1:
        with st.sidebar.popover(':orange[⚠️ 경고 알림]', use_container_width=True):
            warning = """
            SQL 데이터베이스를 대상으로 한 Q&A 시스템을 구축하려면,
            모델이 생성한 SQL 쿼리를 실행해야 합니다. 이를 수행하는 과정에서 
            본질적인 위험이 따릅니다. 데이터베이스 연결 권한은 반드시 
            체인/에이전트의 필요에 맞게 최소한으로 설정하세요.
            \n\n일반적인 보안 모범 사례에 대해 자세히 알아보려면 
            - [여기를 참고하세요](https://python.langchain.com/docs/security).
            """
            st.warning(warning)
        db_uri = st.sidebar.text_input(
            label='데이터베이스 URI',
            placeholder='mysql://user:pass@hostname:port/db'
        )

    elif radio_opt.index(selected_opt) == 2:
        uploaded_file = st.file_uploader("Xls, CSV 등 1개 파일 가능")
        if uploaded_file is not None:
            # bytes_data = uploaded_file.getvalue()
            # st.write(bytes_data)
            # logging.info(f"파일 이름 : {uploaded_file.name}")
            if uploaded_file.name.endswith(("xls", "xlsx")):    
                uploaded_df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith("csv"):    
                uploaded_df = pd.read_csv(uploaded_file)

            # db_file = "temp_database.db"
            db_file = os.path.abspath("../QA-CHATBOT/sample/temp_database.db")
            db_uri = f"sqlite:///{db_file.split("/")[-1]}"
            # db_uri = f"sqlite:///{db_file}"
            logger.info(db_uri)
            
            st.session_state["uploaded_df"] = uploaded_df
            st.session_state["uploaded_file_name"] = uploaded_file.name


    process = st.button("옵션 선택 완료")

    if process and ("process" not in st.session_state):
        st.session_state["process"] = process
        st.session_state["selected_opt"] = radio_opt.index(selected_opt)
        st.session_state["uploaded_file_name"] = sample_db_file
    else:
        st.session_state['selected_opt'] = -1

    
    if not db_uri:
        st.error("Please enter database URI to continue!")
        st.stop()

    clear_db = st.button("공장 초기화", type="primary")

    if clear_db:
        st.session_state["selected_opt"] = -1
        st.session_state["uploaded_file_name"] = "No DB or Files are linked"

print(f"{st.session_state.selected_opt}")

if process:
    if st.session_state.selected_opt == 0:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)    
        render_er(db_uri, "./data/erd.png", mode="auto")
        conn.close()   
        logger.info("샘플 DB 접속 성공 & ERD 생성 성공")

    elif st.session_state.selected_opt == 1:
        conn = sqlite3.connect(db_uri, uri=True)    
        conn.close()        
        logger.info("사용자 DB 접속 성공")
 
        pass
    elif st.session_state.selected_opt == 2:
        logger.info(f"{db_file}입니다.")
        conn = sqlite3.connect(f"{db_file}")
        logger.info("파일 DB 접속 성공")
        uploaded_df.to_sql('uploaded_data', conn, if_exists="replace", index = False)
        conn.close()
    
    if "db_file" not in st.session_state:
        st.session_state["db_file"] = db_file

    if "database" not in st.session_state:
        st.session_state["database"] = SQLDatabase.from_uri(db_uri)
    
    if "agent" not in st.session_state:
        st.session_state["agent"] = create_sql_agent(
                                        llm = llm,
                                        db = st.session_state.database,
                                        # prompt=prompt,
                                        verbose = True,
                                        handle_parsing_errors=True,
                                        handle_sql_errors = True,
                                    )
    # if "chain" not in st.session_state:
    #     st.session_state["chain"] = create_sql_query_chain(
    #                                 llm, 
    #                                 st.session_state.database
    #                                 ) 
                                    
with tab1:
    # if ("process" in st.session_state) and ("uploaded_file_name" in st.session_state) and ("uploaded_df" in st.session_state):
    if st.session_state['selected_opt'] == -1:
        st.title(st.session_state["uploaded_file_name"])
    else:
        if st.session_state.get("process"):
            logger.info(f"Process 상태 : {st.session_state["process"]}")
            logger.info(f"selected opt 번호 : {st.session_state.selected_opt}, 타입 : {type(st.session_state.selected_opt)}")
            if st.session_state.selected_opt == 0:
                st.title(st.session_state["uploaded_file_name"])
                st.image('./data/erd.png')
                pass
            elif st.session_state.selected_opt == 1:
                pass
            elif st.session_state.selected_opt == 2:
                st.title(st.session_state["uploaded_file_name"])
                st.dataframe(st.session_state["uploaded_df"])


with tab2:
    if query := st.chat_input("질문을 입력해주세요."):
        # query = "매출이 많은 10개 주는 어디인가요?"
        agent_executor = st.session_state.agent
        # logger.info(f"### SQL Agent의 프롬프트\n {agent_executor.get_prompts}")
        # chain = st.session_state.chain
        db_file = st.session_state.db_file
        st.chat_message("user").write(query)
        with st.chat_message('assistant'):
            with st.spinner("SQL 쿼리문 생성중..."):
                handler = SQLHandler()
                llm_response = agent_executor.invoke(
                    {"input":query},
                    {"callbacks" : [handler]}
                    )
                generated_query = handler.sql_result
            st.code(
                f'''{generated_query}''', language='sql')
            with st.spinner("SQL 쿼리 실행중..."):
                time.sleep(3)
                # generated_query = chain.invoke({"question":query})
                # logger.info(f"### 생성된 쿼리 : {generated_query}")
                # generated_query = generated_query.split("SQLQuery: ")[-1]
                # st.chat_message('assistant').write(
                #     generated_query)
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                try:
                    cursor.execute(generated_query)
                    column_names = [description[0] for description in cursor.description]
                    query_result = cursor.fetchall()
                except:
                    column_names = None
                    query_result = None
                
                queried_df = pd.DataFrame(data=query_result, columns=column_names)
                conn.close()
        
            st.write(llm_response.get("output"))
            st.dataframe(queried_df)

with tab3:
    pass
    # st.header("An owl")
    # st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

