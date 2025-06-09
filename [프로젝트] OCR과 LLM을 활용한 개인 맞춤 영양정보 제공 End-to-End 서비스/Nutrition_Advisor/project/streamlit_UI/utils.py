import streamlit as st
import os

def config():
    # TODO (후순위) 쿠키값 자바스크립트 실행시켜서 받아오는 것 구현
    # bard api 토큰(cookie 값)
    os.environ['_BARD_API_KEY']= 'cQhAGICCU7iK6XY66rkQ8zYYjtTyY7POBtO77afL_XNVZ7bOKTkBRuw5mFSX9r55EtvrYw.'

def calculate_calories(gender, age, height, weight, activeness):
    calories = 0
    if age <= 17 :
        if gender == '남' :
            if activeness == '활동적':
                calories = 1.9 * (17.7 * weight + 657)
            elif activeness == '보통':
                calories = 1.7 * (17.7 * weight + 657)
            elif activeness == '비활동적':
                calories = 1.4 * (17.7 * weight + 657)

        else :
            if activeness == '활동적':
                calories = 1.8 * (13.4 * weight + 692)
            elif activeness == '보통':
                calories = 1.6 * (13.4 * weight + 692)
            elif activeness == '비활동적':
                calories = 1.4 * (13.4 * weight + 692)

    elif age <= 29 :
        if gender == '남' :
            if activeness == '활동적':
                calories = 1.9 * (15.1 * weight + 692)
            elif activeness == '보통':
                calories = 1.7 * (15.1 * weight + 692)
            elif activeness == '비활동적':
                calories = 1.4 * (15.1 * weight + 692)

        else :
            if activeness == '활동적':
                calories = 1.8 * (14.8 * weight + 487)
            elif activeness == '보통':
                calories = 1.6 * (14.8* weight + 487)
            elif activeness == '비활동적':
                calories = 1.4 * (14.8 * weight + 487)

    else :
        if gender == '남' :
            if activeness == '활동적':
                calories = 1.9 * (11.5 * weight + 873)
            elif activeness == '보통':
                calories = 1.7 * (11.5 * weight + 873)
            elif activeness == '비활동적':
                calories = 1.4 * (11.5 * weight + 873)

        else :
            if activeness == '활동적':
                calories = 1.8 * (8.3 * weight + 846)
            elif activeness == '보통':
                calories = 1.6 * (8.3 * weight + 846)
            elif activeness == '비활동적':
                calories = 1.4 * (8.3 * weight + 846)
    
    return calories

def clear_messages():
    if "messages" in st.session_state:
        st.session_state.messages = []