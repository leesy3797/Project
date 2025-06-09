from prefect import task, flow
from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta, datetime
import time
from crawl_bizinfo_final import craw_bizinfo
from crawl_mss_final import crawl_mss
from filtering_long_list import filtering_long_list
from filtering_short_list import filtering_short_list

# 각 스크립트를 실행하는 Task 정의

@flow(name="Fianl-Workflow")
def crawl_and_filter_workflow():
    craw_bizinfo()
    time.sleep(60)
    crawl_mss()
    time.sleep(60)
    filtering_long_list()
    time.sleep(60)
    filtering_short_list()

if __name__ == '__main__':
  crawl_and_filter_workflow.serve(
    name="Gov-Agent-Deploy",
    schedules=[
      IntervalSchedule(
        interval=timedelta(days=2),
        anchor_date = datetime(2024, 10, 20, 1, 0),
        timezone="Asia/Seoul"
      )
    ]
  )