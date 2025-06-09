import streamlit as st
# from gtts import gTTS
from io import BytesIO

# TODO (후순위) 텍스트 읽어주는 기능 추가
def render():
    message = st.chat_message("assistant")
    message.write(st.session_state.api_response[-1])
    # tts = gTTS(st.session_state.api_response[-1], lang='en')
    # st.audio(BytesIO(tts.get_bytes()), format='audio/mp3')

    if st.button("다음으로"):
        st.session_state.page = 'result_2'