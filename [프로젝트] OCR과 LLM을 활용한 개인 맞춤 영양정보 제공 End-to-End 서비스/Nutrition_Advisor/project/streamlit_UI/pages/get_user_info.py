import streamlit as st
from utils import calculate_calories

def render():
    st.title('정보를 입력해주세요')
    
    user_name = st.text_input('이름', value="User")
    gender = st.selectbox('성별', ['남', '여'])
    height = st.number_input('키(cm 단위로 숫자만 입력해주세요)', min_value=0, value=165)
    weight = st.number_input('몸무게(kg 단위로 숫자만 입력해주세요)', min_value=0, value=65)
    age = st.number_input('나이(숫자로 입력해주세요)', min_value=0, value=25)
    activeness = st.selectbox('활동성', ['활동적', '보통', '비활동적'])

    if 'user_info' not in st.session_state:
        st.session_state.user_info = dict()

    st.session_state.user_info['user_name'] = user_name
    st.session_state.user_info['gender'] = gender
    st.session_state.user_info['height'] = height
    st.session_state.user_info['weight'] = weight
    st.session_state.user_info['calories'] = calculate_calories(gender, age, height, weight, activeness)

    if st.button('다음', key='to_image_upload_option'):
        st.session_state.page = 'image_upload_option'