import streamlit as st
import numpy as np
from PIL import Image

def render():
    st.title('이미지를 업로드 해주세요')

    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])
    if st.button('이미지 사용'):
        if not uploaded_file:
            st.warning('이미지를 업로드 해주세요!')
        else:
            st.session_state.image = np.asarray(Image.open(uploaded_file))
            st.session_state.page = 'img_to_advise'