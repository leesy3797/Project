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

# 체크포인트 불러오기
# @task(name='Load Checkpoint')
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                pass
    return []

# 체크포인트 저장
# @task(name='Save Checkpoint')
def save_checkpoint(data, checkpoint_file):
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 데이터베이스 연결 및 데이터 불러오기
# @task(name='Load Announcements From DB')
def load_announcements_from_db(db_path="crawler/announcement_sqlite.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM announcements")
    data = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    
    cursor.close()
    conn.close()

    # logger.debug("데이터베이스에서 공고문 로드 완료")
    return pd.DataFrame(data=data, columns=columns)

# 공고문을 처리하고 결과를 JSON Lines로 저장
# @task(name='Append Checked Announcement to JsonL')
def append_to_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')

# 파일 다운로드 함수
# @task(name='Downaload Attached File')
def download_file(url, title, save_path="crawler/downloads/attachments"):
    """파일을 지정한 경로에 다운로드하는 함수"""
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # logger.info(url)
    try:
        r = requests.get(url, allow_redirects=True)
    except Exception as e:
        logger.error(e)
        print(e)
        return title
    encoded_file_name = r.headers["Content-Disposition"].split("filename=")[-1].strip('"')
    # decoded_file_name = urllib.parse.unquote(encoded_file_name)
    # file_path = os.path.join(save_path, decoded_file_name)
    file_path = save_path + '/' + title.replace('/', '_') + '.' + encoded_file_name.split('.')[-1]
    with open(file_path, 'wb',) as file:
        file.write(r.content)
    
    logger.info(f"파일 다운로드 완료: {file_path.split("/")[-1]}")
    print(f"파일 다운로드 완료: {file_path.split("/")[-1]}")
    return file_path

# @task
def load_and_split_hwpx(file_path, chunk_size=500, chunk_overlap=100):
    def HWPXLoader(file_path):
        """HWPX 파일에서 문서 구조를 유지하며 본문 텍스트를 추출하는 함수"""
        with zipfile.ZipFile(file_path, 'r') as hwpx:
            for file_info in hwpx.infolist():
                if 'Contents/section0.xml' in file_info.filename:
                    # XML 파일 열기
                    with hwpx.open(file_info) as xml_file:
                        # XML 파싱
                        tree = ET.parse(xml_file)
                        root = tree.getroot()
                        body_text = []
                        # 문단(paragraph)을 기준으로 텍스트 추출
                        for para in root.iter():
                            if para.tag.endswith('}p'):  # 문단을 나타내는 태그
                                paragraph_text = []
                                for elem in para.iter():
                                    if elem.tag.endswith('}t') and elem.text:  # 텍스트 태그
                                        paragraph_text.append(elem.text)
                                # 문단이 완료되면 텍스트를 합쳐서 저장 (문단 단위)
                                if paragraph_text:
                                    body_text.append(' '.join(paragraph_text))
        # 문단을 줄바꿈('\n')으로 구분하여 출력
        return '\n'.join(body_text)
 
    body_text = HWPXLoader(file_path)
    """HWP 파일을 로드하고 텍스트를 분할하는 함수"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_text(body_text)
    docs = [Document(page_content=text) for text in texts]    
    # logger.info(f"HWP 분할 완료: {file_path}")

    return docs   

class HWPLoader(BaseLoader):
    """HWP 파일 읽기 클래스. HWP 파일의 내용을 읽습니다."""

    def __init__(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.extra_info = {"source": file_path}
        self._initialize_constants()

    def _initialize_constants(self) -> None:
        """상수 초기화 메서드"""
        self.FILE_HEADER_SECTION = "FileHeader"
        self.HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
        self.SECTION_NAME_LENGTH = len("Section")
        self.BODYTEXT_SECTION = "BodyText"
        self.HWP_TEXT_TAGS = [67]

    def lazy_load(self) -> Iterator[Document]:
        """HWP 파일에서 데이터를 로드하고 표를 추출합니다.

        Yields:
            Document: 추출된 문서
        """
        load_file = olefile.OleFileIO(self.file_path)
        file_dir = load_file.listdir()

        if not self._is_valid_hwp(file_dir):
            raise ValueError("유효하지 않은 HWP 파일입니다.")

        result_text = self._extract_text(load_file, file_dir)
        yield self._create_document(text=result_text, extra_info=self.extra_info)

    def _is_valid_hwp(self, dirs: List[List[str]]) -> bool:
        """HWP 파일의 유효성을 검사합니다."""
        return [self.FILE_HEADER_SECTION] in dirs and [self.HWP_SUMMARY_SECTION] in dirs

    def _get_body_sections(self, dirs: List[List[str]]) -> List[str]:
        """본문 섹션 목록을 반환합니다."""
        section_numbers = [
            int(d[1][self.SECTION_NAME_LENGTH :])
            for d in dirs
            if d[0] == self.BODYTEXT_SECTION
        ]
        return [
            f"{self.BODYTEXT_SECTION}/Section{num}" for num in sorted(section_numbers)
        ]

    def _create_document(
        self, text: str, extra_info: Optional[Dict] = None
    ) -> Document:
        """문서 객체를 생성합니다."""
        return Document(page_content=text, metadata=extra_info or {})

    def _extract_text(
        self, load_file: olefile.OleFileIO, file_dir: List[List[str]]
    ) -> str:
        """모든 섹션에서 텍스트를 추출합니다."""
        sections = self._get_body_sections(file_dir)
        return "\n".join(
            self._get_text_from_section(load_file, section) for section in sections
        )

    def _is_compressed(self, load_file: olefile.OleFileIO) -> bool:
        """파일이 압축되었는지 확인합니다."""
        with load_file.openstream(self.FILE_HEADER_SECTION) as header:
            header_data = header.read()
            return bool(header_data[36] & 1)

    def _get_text_from_section(self, load_file: olefile.OleFileIO, section: str) -> str:
        """특정 섹션에서 텍스트를 추출합니다."""
        with load_file.openstream(section) as bodytext:
            data = bodytext.read()

        unpacked_data = (
            zlib.decompress(data, -15) if self._is_compressed(load_file) else data
        )

        text = []
        i = 0
        while i < len(unpacked_data):
            header, rec_type, rec_len = self._parse_record_header(
                unpacked_data[i : i + 4]
            )
            if rec_type in self.HWP_TEXT_TAGS:
                rec_data = unpacked_data[i + 4 : i + 4 + rec_len]
                text.append(rec_data.decode("utf-16"))
            i += 4 + rec_len

        text = "\n".join(text)
        text = self.remove_chinese_characters(text)
        text = self.remove_control_characters(text)
        return text

    @staticmethod
    def remove_chinese_characters(s: str):
        """중국어 문자를 제거합니다."""
        return re.sub(r"[\u4e00-\u9fff]+", "", s)

    @staticmethod
    def remove_control_characters(s):
        """깨지는 문자 제거"""
        return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

    @staticmethod
    def _parse_record_header(header_bytes: bytes) -> tuple:
        """레코드 헤더를 파싱합니다."""
        header = struct.unpack_from("<I", header_bytes)[0]
        rec_type = header & 0x3FF
        rec_len = (header >> 20) & 0xFFF
        return header, rec_type, rec_len 


# PDF 로드 및 텍스트 분할
# @task
def load_and_split_pdf(file_path, chunk_size=500, chunk_overlap=100):
    """PDF 파일을 로드하고 텍스트를 분할하는 함수"""
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # logger.info(f"PDF 분할 완료: {file_path}")
    return text_splitter.split_documents(docs)

# 벡터스토어 생성
# @task
def create_vector_store(texts):
    """문서 텍스트를 벡터로 변환하여 벡터 스토어 생성"""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # logger.info("벡터 스토어 생성 완료")
    if (not texts) or (type(texts[0]) == str):
        return FAISS.from_texts(texts=['세부 정보 없음'], embedding=embeddings)
    else:
        return FAISS.from_documents(documents=texts, embedding=embeddings)
    
# 프롬프트 설정
# @task
def generate_prompt_template():
    """프롬프트 템플릿 설정"""
    # logger.info("프롬프트 템플릿 생성 완료")
    return PromptTemplate(
        input_variables=["title", "context"],
        template=(
            """당신은 AI 기반 초자동화 서비스를 제공하는 이커머스(ecommerce) 유통 도메인의 기업을 지원하는 컨설턴트입니다. 
            이 회사는 'AI Agent' 기술 개발과 사업화를 위한 정부의 R&D 및 사업화 지원이 필요합니다. 
            당신의 역할은 주어진 정부 지원 과제 공고문을 분석하여, 이 회사에 적합한지 여부를 평가한 후, 결과를 양식에 맞춰 출력하는 것입니다.
           
            이 작업을 단계별로 나누어 진행할 것입니다. 아래 각 단계를 차례로 수행해 주세요.

            ### Step 1: 공고문 요약
            아래 주어진 context를 참고해서 공고문을 요약해 주세요. 
            요약에는 다음 항목을 포함해야 합니다:
            - 공고 제목
            - 주요 목표 및 내용 (가능한 주요 내용이 누락되지 않게 상세히 작성)
            - 지원 대상
            - 지원 금액 (가능한 경우)
            - 지원 기간 (가능한 경우)

            ### Step 2: 평가 기준에 따른 평가
            요약된 공고문을 바탕으로 아래의 평가 기준에 따라 해당 공고가 회사에 적합한지 평가해 주세요.
            
            평가 기준:
            1. 공고문이 AI 기반 초자동화 기술과 관련이 있는가? (반드시 초자동화 기술이 아니어도 됩니다. AI 또는 빅데이터 등 광범위한 영역에서 평가해주세요.)
            2. 해당 공고가 R&D 지원을 제공하는가? (R&D 지원을 필수로 하되, 이외에도 기업 운영과 기술 개발에 도움이 될만한 지원 내용이 있으면 해당 공고를 고려해주세요.)
            3. 해당 공고가 사업화 지원을 제공하는가? (금전적 지원 외에도 기업의 기술 개발과 기업 경영을 지원하는 내용이 있으면 해당 공고를 고려해주세요.)
            4. 공고문의 지원 대상이 회사와 적합한가? (지역적으로 서울을 소재의 기업을 대상으로 한 사업이어야 하며, 3년차 스타트업이 지원 받을 수 있는 내용인지 확인해주세요.)

            ### Step 3: 최종 결과 출력
            평가가 완료되면, context 내용을 참고해서 다음 양식에 맞춰 최종 결과를 Json 형식으로 출력해 주세요. 
            만약 context에 아무런 내용이 없거나 '세부 정보 없음'이라고 되어 있을 경우,
            결과 Attribute는 '보류'로 표기하고, 나머지 Attribute들에 대해서는 Value를 '세부 정보 없음'이라고 채워줘. 
            OUTPUT FORMAT MUST BE JSON!!!! 
            
            - 결과: 추천 / 비추천 중 선택
            - 사업 공고명: ${title}$ (사업 공고명은 $와 $사이의 텍스트만 그대로 가져와줘)
            - 사업 형식 : 사업 형태 ('사업화 지원' 또는 'R&D 지원' 중 적합한 것으로 선택)
            - 이유: 해당 공고를 추천하면 적합한 이유를, 비추천한다면 적합하지 않은 이유에 대해서 설명
            - 사업 기간: 제공된 사업 기간 (가능한 경우)
            - 지원 금액: 제공된 지원 금액 (가능한 경우)
            - 사업 개요(요약): 공고문 요약 내용
            - 기타 중요 정보: 기타 중요한 정보 (가능한 경우)

            Context:
            {context}

            Answer:"""
        )
    )

# 공고문 평가 함수
# @task
def evaluate_announcement(title, retriever, llm):
    """공고문을 평가하고 결과를 출력"""
    # 컨텐츠 결합 함수
    def content_join(retrieved_docs):
        return "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 프롬프트 템플릿 설정
    prompt_template = generate_prompt_template()

    query = "사업 공고문의 사업 목적, 신청 자격, 지원 내용, 지원 조건 등에 대해서 요약 정리해주세요."
    # 체인 설정 및 실행
    chain = (
        {   
            "title": RunnablePassthrough(),
            "context": RunnablePassthrough() | (lambda x: query) | retriever | content_join
        }
        | prompt_template
        | llm
        | JsonOutputParser()
    )
    
    # 결과 출력
    # logger.info(f"공고문 평가 완료: {title}")
    return chain.invoke(title)