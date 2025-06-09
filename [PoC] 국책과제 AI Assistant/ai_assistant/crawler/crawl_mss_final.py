import os
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from loguru import logger
from crawl_utils import *
# from prefect import task, flow
# from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta, datetime

# 크롤러 함수
# @task(name='Crawling in Mss')
def crawl_announcements(driver, page_limit=12, data_list=[]):
    """웹 페이지를 크롤링하고 데이터를 수집하여 리스트와 DB에 저장하는 함수"""
    base_url = "https://www.mss.go.kr/site/smba/ex/bbs/List.do?cbIdx=310"
    
    # 마지막 실행 날짜 불러오기
    last_execution_date = get_last_execution_date()

    if last_execution_date:
        logger.info(f"마지막 실행 날짜: {last_execution_date}. 그 이후의 공고만 크롤링합니다.")
        print(f"마지막 실행 날짜: {last_execution_date}. 그 이후의 공고만 크롤링합니다.")
    for page in range(1, page_limit + 1):
        driver.get(f"{base_url}&pageIndex={page}")
        time.sleep(2)  # 페이지 로딩 대기
        logger.info(f"현재 {page}/{page_limit} 페이지 크롤링 중...")
        print(f"현재 {page}/{page_limit} 페이지 크롤링 중...")

        # 공고 리스트 추출
        announcements = driver.find_elements(By.CSS_SELECTOR, "#contents_inner > div > table > tbody > tr")
        time.sleep(1.5)
        
        for announcement in announcements:
            try:
                # 각 항목 데이터 추출
                title = announcement.find_element(By.CSS_SELECTOR, "td.subject").text.strip()
                application_period = announcement.find_element(By.CSS_SELECTOR, "td:nth-child(3)").text.strip()
                # announcement_number = announcement.find_element(By.CSS_SELECTOR, "td:nth-child(4)").text.strip()
                department = announcement.find_element(By.CSS_SELECTOR, "td:nth-child(6)").text.strip()
                posted_date = announcement.find_element(By.CSS_SELECTOR, "td:nth-child(7)").text.strip()

                # 날짜 형식 변환
                posted_date_parsed = datetime.strptime(posted_date, "%Y.%m.%d").date()

                # 마지막 실행 날짜 이후의 공고만 크롤링
                if last_execution_date and posted_date_parsed < datetime.strptime(last_execution_date, "%Y-%m-%d").date():
                    logger.info(f"등록일 {posted_date} 이전의 공고는 크롤링 중단.")
                    return data_list  # 중단

                attached_url = announcement.find_element(By.CSS_SELECTOR, "td.attached-files > a").get_attribute('href')
                
                # 상세 URL 추출
                onclick_attr = announcement.find_element(By.CSS_SELECTOR, "td.subject > a").get_attribute("onclick")
                args = onclick_attr.split("'")[1::2]
                cbIdx, bcIdx, parentSeq = args[0], args[1], args[3]
                detail_url = f"https://www.mss.go.kr/site/smba/ex/bbs/View.do?cbIdx={cbIdx}&bcIdx={bcIdx}&parentSeq={parentSeq}"

                # 상세 페이지로 이동하여 세부 내용 크롤링
                driver.get(detail_url)
                time.sleep(2)  # 페이지 로딩 대기
                iframe = driver.find_element(By.CSS_SELECTOR, "iframe.iframeRes")
                driver.switch_to.frame(iframe)

                # iframe 내의 콘텐츠 크롤링
                description = driver.find_element(By.CSS_SELECTOR, "body").text.strip()
                
                # 다시 메인 페이지로 돌아오기
                driver.switch_to.default_content()
                # 페이지 스크롤 처리
                for i in range(2):
                    driver.execute_script("window.scrollBy(0, 500);")
                    time.sleep(1)

                driver.back()
                time.sleep(2)

                # 데이터 리스트에 추가
                data_list.append({
                    "제목": title,
                    "신청기간": application_period,
                    # "공고번호": announcement_number,
                    "담당부서": department,
                    "첨부파일": attached_url,
                    "세부 URL": detail_url,
                    "등록일": posted_date,
                    "설명": description
                })
                
                logger.info(f"공고 크롤링 완료: {title}")
                print(f"공고 크롤링 완료: {title}")

                # 데이터베이스에 삽입
                data = (title, application_period, department, attached_url, detail_url, posted_date, description)
                insert_announcement(data)

            except Exception as e:
                logger.error(f"에러 발생: {e}")
                print(f"에러 발생: {e}")
                continue
    
    return data_list

# 메인 함수
# @flow
def crawl_mss():
    """크롤링을 실행하고 결과를 엑셀 파일로 저장하는 메인 함수"""
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # 브라우저를 숨기고 백엔드에서 실행
    # options.add_argument("--disable-gpu")  # GPU 비활성화 (일부 환경에서 필요)
    # options.add_argument("--no-sandbox")  # 샌드박스 모드 비활성화 (Linux 환경에서 필요)
    # options.add_argument("--window-size=1920x1080")  # 창 크기를 설정하여 페이지가 잘 렌더링되도록 함
    # options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager(latest_release_url="https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE").install()), options=options)

    # 테이블 생성
    create_table()
    
    # 크롤링 시작
    data_list = crawl_announcements(driver)
    driver.quit()

    # 결과를 엑셀 파일로 저장
    if not os.path.exists("downloads"):
        os.makedirs("downloads")
    
    df = pd.DataFrame(data_list)
    df.to_excel("downloads/mss_announcements.xlsx", index=False)
    logger.info("크롤링 결과가 'downloads/mss_announcements.xlsx' 파일로 저장되었습니다.")
    print("크롤링 결과가 'downloads/mss_announcements.xlsx' 파일로 저장되었습니다.")
    
    # 마지막 실행 날짜 저장
    last_execution_date = set_last_execution_date()
    logger.info("마지막 실행 날짜가 저장되었습니다.")
    print("마지막 실행 날짜가 저장되었습니다.")
    
    st_message = f'시작 %% {last_execution_date}까지 총 {len(data_list)}건 수집 완료 %% 끝'
    print(st_message)

# if __name__=='__main__':
#     crawl_mss.serve(
#   name="crawl_mss",
#   schedules=[
#     IntervalSchedule(
#       interval=timedelta(days=2), 
#       anchor_date = datetime(2024, 10, 20, 1, 0),
#       timezone="Asia/Tokyo",
#     )
#   ]
# )

if __name__=="__main__":
    crawl_mss()