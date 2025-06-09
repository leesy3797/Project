import os
import pandas as pd
from loguru import logger
import sqlite3
from datetime import datetime
# from prefect import task, flow

# SQLite 연결 함수
# @task(name='Connect DB')
def connect_db():
    """SQLite 데이터베이스에 연결하는 함수"""
    return sqlite3.connect("crawler/announcement_sqlite.db")

# 테이블 생성 함수
# @task(name='Create Table in DB', log_prints=True)
def create_table():
    """announcements 테이블을 생성하는 함수, 테이블이 없을 경우 생성"""
    conn = connect_db()
    cursor = conn.cursor()
    
    # 테이블 존재 여부 확인
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='announcements';")
    table_exists = cursor.fetchone()
    
    if table_exists:
        logger.info("데이터베이스 테이블 'announcements' 이미 존재함.")
        print("데이터베이스 테이블 'announcements' 이미 존재함.")
    else:
        cursor.execute('''
            CREATE TABLE announcements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title VARCHAR(255),
                application_period VARCHAR(255),
                announcement_number VARCHAR(100),
                department VARCHAR(255),
                attachment VARCHAR(255),
                url VARCHAR(255) UNIQUE,
                posted_date DATE,
                description TEXT
            )
        ''')
        logger.info("데이터베이스 테이블 'announcements' 생성 완료.")
        print("데이터베이스 테이블 'announcements' 생성 완료.")
    conn.commit()
    cursor.close()
    conn.close()

# 크롤링된 데이터 DB에 밀어넣기(중복 무시)
# @task(name='Insert Data into DB', log_prints=True)
def insert_announcement(data):
    """크롤링한 데이터를 SQLite 데이터베이스에 삽입하는 함수, 중복 시 무시"""
    conn = connect_db()
    cursor = conn.cursor()
    query = '''
        INSERT OR IGNORE INTO announcements 
        (title, application_period, department, attachment, url, posted_date, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    cursor.execute(query, data)
    
    if cursor.rowcount == 0:
        logger.info(f"중복된 데이터로 인해 삽입 무시: {data[0]}")
        print(f"중복된 데이터로 인해 삽입 무시: {data[0]}")
    else:
        logger.info(f"새로운 데이터 삽입 완료: {data[0]}")
        print(f"새로운 데이터 삽입 완료: {data[0]}")
    
    conn.commit()
    cursor.close()
    conn.close()

# 마지막 실행 날짜 불러오기
# @task(name='Get Last Execution Date')
def get_last_execution_date():
    """마지막 실행 날짜를 불러오는 함수"""
    if os.path.exists("crawler/resources/last_execution_date.txt"):
        with open("crawler/resources/last_execution_date.txt", "r") as f:
            return f.read().strip()
    return None

# 마지막 실행 날짜 저장
# @task(name='Set Last Execution Date')
def set_last_execution_date():
    """마지막 실행 날짜를 저장하는 함수"""
    with open("crawler/resources/last_execution_date.txt", "w") as f:
        f.write(datetime.today().strftime('%Y-%m-%d'))
    return datetime.today().strftime('%Y-%m-%d')