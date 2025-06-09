import requests
import time
from bs4 import BeautifulSoup
import os
import pandas as pd
from loguru import logger
from datetime import datetime
from crawl_utils import *
# from prefect import task, flow
# from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta, datetime

# @flow(name='crawling_bizinfo', log_prints=True)
def craw_bizinfo():
    # 수집할 데이터를 저장할 리스트
    data_list = []

    # 베이스 URL 및 헤더 설정
    base_url = 'https://www.bizinfo.go.kr/web/lay1/bbs/S1T122C128/AS/74/'
    list_url = base_url + 'list.do'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
    }

    create_table()

    last_execution_date = get_last_execution_date()

    if last_execution_date:
        logger.info(f"마지막 실행 날짜: {last_execution_date}. 그 이후의 공고만 크롤링합니다.")
        print(f"마지막 실행 날짜: {last_execution_date}. 그 이후의 공고만 크롤링합니다.")
    EXIT_FLAG = False

    # 페이지 순회하며 데이터 수집
    for page_num in range(1,100):  # 여기서는 5페이지까지로 설정했지만 원하는 페이지 수로 조정 가능
        if EXIT_FLAG:
            break
        params = {
            'rows': 15,  # 한 페이지당 15개의 공고 표시
            'cpage': page_num  # 페이지 번호
        }
        response = requests.get(list_url, headers=headers, params=params)
        time.sleep(2)
        
        if response.status_code != 200:
            # logger.info(f"Error fetching page {page_num}: {response.status_code}")
            print(f"Error fetching page {page_num}: {response.status_code}")
            continue
        logger.info(f"{page_num}페이지 접속 성공")
        print(f"{page_num}페이지 접속 성공")
        
        # BeautifulSoup으로 페이지 파싱
        soup = BeautifulSoup(response.text, "html.parser")
        urls = soup.find_all("td", class_="txt_l")
        
        url_num = 1

        for url in urls:
            link = url.find('a').get('href')
            detail_url = os.path.join(base_url, link)
            response = requests.get(detail_url, headers=headers)
            if response.status_code != 200:
                logger.info(f"Error fetching detail page: {response.status_code}")
                print(f"Error fetching detail page: {response.status_code}")
                continue
            logger.info(f"{url_num}번째 공고문 접속 성공")
            print(f"{url_num}번째 공고문 접속 성공")
            time.sleep(2)
            
            # 상세 페이지 파싱
            detail_soup = BeautifulSoup(response.text, "html.parser")
            title = detail_soup.find("h2", class_="title").get_text().strip()  # 공고 제목
            
            info_fields = detail_soup.find_all("div", class_="txt")
            department = info_fields[1].get_text().strip()  # 사업 수행 기관
            application_period = ' '.join(info_fields[2].get_text().strip().split()).replace(".", "-")  # 신청 기간 (공백 제거)
            description = info_fields[3].get_text().strip()  # 사업 개요
            posted_date = detail_soup.find("div", class_='top_info').get_text().strip().split("\r")[0]  # 공고일
            
            # 날짜 형식 변환
            posted_date_parsed = datetime.strptime(posted_date, "%Y.%m.%d").date()

            # 마지막 실행 날짜 이후의 공고만 크롤링
            if last_execution_date and posted_date_parsed < datetime.strptime(last_execution_date, "%Y-%m-%d").date():
                logger.info(f"등록일 {posted_date} 이전의 공고는 크롤링 중단.")
                print(f"등록일 {posted_date} 이전의 공고는 크롤링 중단.")
                
                EXIT_FLAG = True # 중단
                break

            # 첨부 파일 URL 추출 (없을 수도 있음)
            try:
                attached_url = "https://www.bizinfo.go.kr" + detail_soup.find_all("a", class_="basic-btn01 btn-gray-bd icon_download")[-1].get("href")
            except IndexError:
                attached_url = "첨부 파일 없음"
            
            # 수집한 데이터를 리스트에 추가
            data_list.append({
                "제목": title,
                "신청기간": application_period,
                "담당부서": department,
                "첨부파일": attached_url,
                "세부 URL": detail_url,
                "등록일": posted_date,
                "설명": description
            })
            
            # 데이터베이스에 삽입
            data = (title, application_period, department, attached_url, detail_url, posted_date, description)
            insert_announcement(data)
            
            url_num += 1

    # pandas를 이용해 Excel로 저장
    df = pd.DataFrame(data_list)
    df.to_excel('downloads/bizinfo_announcement.xlsx', index=False)
    logger.info("크롤링 완료 및 Excel 저장 완료: bizinfo_announcement.xlsx")
    print("크롤링 완료 및 Excel 저장 완료: bizinfo_announcement.xlsx")

    st_message = f'시작 %% {last_execution_date}부터 총 {len(data_list)}건 수집 완료 %% 끝'
    print(st_message)

# if __name__=='__main__':
#     craw_bizinfo.serve(
#   name="crawl_bizinfo",
#   schedules=[
#     IntervalSchedule(
#       interval=timedelta(days=2),
#       anchor_date = datetime(2024, 10, 20, 1, 0),
#       timezone="Asia/Seoul",
#     )
#   ]
# )
    
if __name__ == '__main__':
    craw_bizinfo()
