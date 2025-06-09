import streamlit as st
from utils import config
from pages import get_user_info
from pages import image_upload_option
from pages import image_upload
from pages import camera
from pages import img_to_advise_paddle
from pages import result_1
from pages import result_2

config()

# 페이지 이동 처리
if 'page' not in st.session_state:
    st.session_state.page = 'get_user_info'
if st.session_state.page == 'get_user_info':
    get_user_info.render()
elif st.session_state.page == 'image_upload_option':
    image_upload_option.render()
elif st.session_state.page == 'image_upload':
    image_upload.render()
elif st.session_state.page == 'camera':
    camera.render()
elif st.session_state.page == 'img_to_advise':
    img_to_advise_paddle.render()
elif st.session_state.page == 'result_1':
    result_1.render()
elif st.session_state.page == 'result_2':
    result_2.render()