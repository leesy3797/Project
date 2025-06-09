import os
import urllib
import json
import requests
from dotenv import load_dotenv
import pandas as pd
import time
import warnings
from loguru import logger
from tqdm import tqdm
import sqlite3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from google import generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Iterator
import olefile
import zlib
import struct
import re
import unicodedata
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
# from prefect import *
from filter_utils import *

# 경고 무시 설정
warnings.filterwarnings('ignore')

# 전체 프로세스 실행
# @task(name='Check Announcements', log_prints=True)
def process_announcements(announcement_df, long_list_df, jsonl_file, llm):
    """전체 공고문 평가 프로세스를 실행하는 함수"""

    checkpoint_file = "crawler/resources/checkpoint.json"
    complete_titles = load_checkpoint(checkpoint_file)
    # 공고문과 첨부 파일을 병합한 데이터프레임
    merge_df = pd.merge(long_list_df, announcement_df, how='inner', on="title")
    tmp_df = merge_df[["title", "attachment"]]
    
    logger.info(f"총 {len(tmp_df)}개의 공고문을 처리합니다.")
    print(f"총 {len(tmp_df)}개의 공고문을 처리합니다.")
    
    # 각 공고문 처리
    # for idx, row in tmp_df.iterrows():
    for idx, row in tqdm(tmp_df.iterrows(), total=len(tmp_df), desc="Short List 추출 진행"):
        title = row["title"]
        if title in complete_titles:
            logger.info(f"[PASS] {title}")
            print(f"[PASS] {title}")
            continue        
        try:
            file_path = download_file(row["attachment"], title)
            if file_path.endswith(".pdf"):
                # PDF 로드 및 분할
                texts = load_and_split_pdf(file_path)
            elif file_path.endswith(".hwpx"):
                texts = load_and_split_hwpx(file_path)

            elif file_path.endswith(".hwp"):
                loader = HWPLoader(file_path)
                texts = loader.load_and_split()
            else:
                texts = ["세부 정보 없음"]
                response = {"결과": "보류", "사업 공고명": f"{title}", "사업 형식": "상세 내용 없음", "이유": "상세 내용 없음", "사업 기간": "상세 내용 없음", "지원 금액": "상세 내용 없음", "사업 개요(요약)": "상세 내용 없음", "기타 중요 정보": "상세 내용 없음"}
                append_to_jsonl(jsonl_file, response)
                
                logger.info(f"{idx+1}번째 공고문 평가 결과: {response["결과"]}")
                print(f"{idx+1}번째 공고문 평가 결과: {response["결과"]}")

                complete_titles.append(title)
                save_checkpoint(complete_titles, checkpoint_file)
                continue

            # logger.debug(texts)
            # 벡터스토어 생성
            vector_store = create_vector_store(texts)
            retriever = vector_store.as_retriever(search_type='mmr', search_kwargs = {"k": 10, "fetch_k" : 50})
            
            # 공고문 평가
            try:
                response = evaluate_announcement(title, retriever, llm)
            except:
                response = {"결과": "보류", "사업 공고명": f"{title}", "사업 형식": "상세 내용 없음", "이유": "상세 내용 없음", "사업 기간": "상세 내용 없음", "지원 금액": "상세 내용 없음", "사업 개요(요약)": "상세 내용 없음", "기타 중요 정보": "상세 내용 없음"}

            append_to_jsonl(jsonl_file, response)
            logger.info(f"{idx+1}번째 공고문 평가 결과: {response["결과"]}")
            time.sleep(5)

            complete_titles.append(title)
            save_checkpoint(complete_titles, checkpoint_file)

        except Exception as e:
            logger.error(f"에러 발생 : {e}")
            print(f"에러 발생 : {e}")
            continue

# 메인 함수
# @flow(name='Filter for Short List', log_prints=True)
def filtering_short_list():
        
    # 환경 변수 로드 및 설정
    load_dotenv()
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # LLM 설정
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    # logger.debug("LLM 설정 완료")

    # 데이터베이스에서 공고문 로드
    announcement_df = load_announcements_from_db()
    
    # JSON 파일에서 long list 로드
    long_list_df = pd.read_json("crawler/downloads/long_list.json")
    
    # 공고문 평가 프로세스 실행
    logger.info("Short List 평가 프로세스 시작")
    jsonl_file = "crawler/downloads/short_list.jsonl"
    process_announcements(announcement_df, long_list_df, jsonl_file, llm=llm)
    
    logger.info("Short List 평가 프로세스 종료")
    print(f"시작 %% Short List 평가 프로세스 종료 %% 끝")


if __name__=='__main__':
    filtering_short_list()