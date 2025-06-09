import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google import generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from IPython.display import clear_output
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import time
from datetime import datetime, timedelta
from loguru import logger
import sqlite3
from langchain_core.exceptions import OutputParserException
import json
# from prefect import task, flow



# @task(name='LLM Invoke', log_prints=True)
def llm_invoke(chain, batch_announcement):
    return chain.invoke({"batch":batch_announcement})
    
# @flow(name='Filter for Long List', log_prints=True)
def filtering_long_list():
    load_dotenv()
    genai.configure(api_key = os.environ["GOOGLE_API_KEY"])

    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.5-flash-latest"
    )

    with open("crawler/resources/last_execution_date.txt", 'r') as file:
        last_execution_date = file.read().strip()

    if last_execution_date:
        # 전날 구하기
        previous_day = datetime.strptime(last_execution_date, '%Y-%m-%d') - timedelta(days=15)

        # 전날을 '%Y-%m-%d' 형식의 문자열로 출력
        previous_day = previous_day.strftime('%Y.%m.%d')
        logger.info(f"지난 {previous_day} 이후의 공고만 Agent가 체크합니다.")
        print(f"지난 {previous_day} 이후의 공고만 Agent가 체크합니다.") 

    conn = sqlite3.connect("crawler/announcement_sqlite.db")
    cursor = conn.cursor()
    cursor.execute(f"select * from announcements where posted_date >= '{previous_day}'")

    data = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(data = data, columns=columns)

    cursor.close()
    conn.close()

    logger.info(f"## 총 {df.shape[0]}개의 공고 확인 시작 ##")
    print(f"## 총 {df.shape[0]}개의 공고 확인 시작 ##")

    system_prompt = """당신은 기업들을 대상으로 국가 지원사업을 컨설팅하는 전문 컨설턴트입니다. 
    중소기업이나 스타트업은 자금력이 부족하여 사업을 영위하고, 
    새로운 기술을 개발할 때 정부로부터의 지원이 반드시 필요합니다. 
    하지만, 정부에서 시행하는 지원사입이 너무 많고, 지원 내용이 복잡하다보니,
    대부분의 기업에서는 자신의 기업 상황에 적합한 지원 사업을 찾아내고 신청하는 것이 매우 어려운 실정입니다.
    이에 당신의 역할은 특정 기업의 "사업 내용"과 "새로운 기술 개발" 내용을 이해하고, 
    해당 내용을 수행하는데 적합한 정부 지원 과제를 추천해주는 것입니다.
    정부 과제 공고 리스트를 확인하고, 기업들에게 추천해주시기 바랍니다.
    꼼꼼히 빠트리지 않고 기업에 적합한 과제들을 많이 추천할수록 당신에게는 매 건마다 100만원의 인센티브가 주어집니다."""

    system_message = SystemMessagePromptTemplate.from_template(
        system_prompt
    )
    user_prompt = """
        우리 회사는 이커머스 도메인에서 'AI Agent'를 개발하여 AI 기반의 초자동화 서비스를 제공하는 AI 기업입니다.
        AI 기반의 초자동화 서비스를 개발하고 이를 사업화 하기 위해서는 정부로부터의 "R&D지원"과 "사업화지원"를 받는 것이 매우 중요합니다.
        아래의 정부 과제 공고 목록 중 우리 회사에 적합한(추천하는) 공고들을 판단해 주세요. 
        
        {batch}
        
        최종 답변은 아래 답변 예시처럼 추천하는 공고문의 "공고제목(title)", 
        "추천하는 이유(reason)"를 리스트(list) 안의 JSON 형식으로 출력해주세요.
        공고제목(title)은 위에서 사용된 공고제목을 그대로 가져와주세요. 
        공고제목은 "공고 제목: " 뒤에 있으며, 반드시 공고제목은 대괄호를 제외한 대괄호([]) 안의 문자 그대로 가져오세요.  
        OUTPUT MUST BE IN JSON FORMAT LIKE EXAMPLE BELOW
        (예시) ```json
            [
                {{"title" : "공고제목1", "reason" : "이유1"}}, 
                {{"title" : "공고제목2", "reason" : "이유2"}},
                ...
                {{"title" : "공고제목N", "reason" : "이유N"}}
            ]```     
        """

    tmp = """최종 답변은 아래 답변 예시처럼 인덱스만 출력해주면 돼요. 다른 부가 설명 필요없고, 
        반드시 추천 공고 리스트에 대한 인덱스만 리스트 형식으로 출력해주세요.  
        (예시) [5, 10, 98, 120]"""

    prompt =ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
    )

    chain = (
        prompt 
        | llm
        # | StrOutputParser()
        | JsonOutputParser()
    )

    def get_candidate(idx, title, application_period, description):
        announcement_prompt = f"""
        ######## {idx+1}번째 공고문 ########
        공고 제목: {title}
        신청 기간: {application_period}
        공고 설명: {description}
        """
        return announcement_prompt

    batch_size = 10
    num_row = df.shape[0]
    long_list = []

    for i in range(0, num_row, batch_size):
        logger.info(f"##### {(i/batch_size)+1:.0f}/{np.ceil(num_row/batch_size):.0f} 배치 시작 ######")
        print(f"##### {(i/batch_size)+1:.0f}/{np.ceil(num_row/batch_size):.0f} 배치 시작 ######")
        # clear_output(wait=True)
        # time.sleep(1)
        batch_df = df[i:i+batch_size]
        announcement_list = []
        for idx, row in batch_df.iterrows():
            id, title, application_period, _, _, _, _, description = row
            announcement = get_candidate(id, title, application_period, description)
            announcement_list.append(announcement)
            batch_announcement = "\n".join(announcement_list)
        # logger.info(batch_announcement)
        logger.info(f"1. 성공적으로 Batch구성")
        try:
            response = llm_invoke(chain, batch_announcement)
            # logger.info(response)
            long_list.extend(response)
            time.sleep(5)
            logger.info("2. 추천 필터링 성공")

        except Exception as e:
            logger.info(f"60초 대기:\n {e}\n")
            try:
                time.sleep(20)
                response = response = llm_invoke(chain, batch_announcement)
                # logger.info(response)
                long_list.extend(response)
                logger.info("2. 추천 필터링 성공")

            except json.decoder.JSONDecodeError as e:
                logger.error(f"JSONDecodeError: {e} => 다음 배치로 스킵")
                continue  # 에러가 발생하면 현재 배치b를 건너뛰고 다음 배치로 넘어감

            except OutputParserException as e:
                logger.error(f"OutputParserException: {e} => 다음 배치로 스킵")
                continue  # OutputParserException 발생 시 건너뛰기         

    logger.info(f"## 총 {len(long_list)}개({(len(long_list)/df.shape[0]*100):.2f}%)의 공고 필터링(Long List) ##")
    print(f"시작 %% 총 {len(long_list)}개({(len(long_list)/df.shape[0]*100):.2f}%)의 공고 필터링(Long List) %% 끝")

    # JSON 파일로 저장
    with open('crawler/downloads/long_list.json', 'w', encoding='utf-8') as json_file:
        json.dump(long_list, json_file, ensure_ascii=False, indent=4)

    logger.info("Long List => JSON 파일 저장(덮어쓰기) 완료")
    print("Long List => JSON 파일 저장(덮어쓰기) 완료")

if __name__ == "__main__":
    filtering_long_list()