import streamlit as st
import base64
import image_to_text
from bardapi import Bard
import traceback

# TODO 프롬프트 및 모델 변경사항 반영
def render():
    if 'img_to_advise_done' not in st.session_state:
        st.session_state.img_to_advise_done = False
    
    # 대기 영상 재생
    file_ = open("/Users/jiyoo/ASAC_data_analysis_3rd/nutrition-fact-img-to-advise/resources/spongebob-thinking.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    st.title(f"고민중....")

    if not st.session_state.img_to_advise_done:
        # image -> text -> bard api -> advise
        try:
            print("이미지 텍스트 변환 시도")
            result_text = image_to_text.easy_ocr(st.session_state.image)
        except Exception as e:
            print(f"이미지 텍스트 변환 실패")
            traceback.print_exc()
            st.session_state.page = 'image_upload_option'
        else:
            print("이미지 텍스트 변환 성공")
        
        query = f"""
        너는 세상에서 가장 똑똑한 영양사이자 보고서 작성에 능숙한 마케터야. 아래 텍스트는 식제품의 뒷면 영양정보 및 주의사항이야.
        다소 잡음과 오타가 있는데, 이를 최대한 원본에 가깝게 수정해줘.
        {result_text}
        이후 해당 내용을 바탕으로 
        1) 이 음식 섭취시 주의해야하는 사항들(알레르기, 연관 질병 등), 
        2) 하루 기초대사량이 2500kcal인 사람 기준 영양구성 비중 
        3) 주어진 운동별로 이 칼로리를 소비하기 위해 운동해야하는 시간을 아래의 양식으로 정리해줘.

        1. 이런 분들은 이 음식 섭취에 주의해야 합니다!

        1) 땅콩 알레르기가 있다면 섭취에 주의해야해요.
        2) 유당분리증이 있다면 섭취에 주의해야해요.
        3) 기타 주의사항들
        4) ...

        2. {st.session_state.user_info['user_name']}님의 1일 기초대사량 대비 영양성분 구성은 다음과 같아요.

        1) 총 칼로리 : OOkcal (OO%)
        2) 나트륨 : OOmg (OO%)
        3) 탄수화물 : OOg (OO%)
        4) 당류 : OOg (OO%)
        5) 지방 : OOg (OO%)
        6) 트랜스지방 : OOg
        7) 포화지방 : OOg (OO%)
        8) 콜레스테롤 : OOg (OO%)
        9) 단백질 : 7g OOg (OO%)

        3. {st.session_state.user_info['user_name']}님, 이 음식을 소화하기 위해서는 아래와 같은 운동 시간이 필요해요!

        > 헬스 : 00분
        > 등산 : 00분
        > 축구 : 00분
        > 농구 : 00분 
        > 수영 : 00분
        > 러닝 : 00분
        > 걷기 : 00분
        """
        
        # bard api 호출
        if 'api_response' not in st.session_state:
            st.session_state.api_response = list()
        try:
            print('bard api 호출 시도')
            st.session_state.api_response.append(Bard().get_answer(str(query))['content'])
            print("====================st.session_state.api_response[-1]========================")
            print(st.session_state.api_response[-1])
            print("=========================================================================")
        except Exception as e:
            print(f"bard api 호출 실패")
            traceback.print_exc()
            st.warning('bard api 호출 실패!')
            exit()
        else:
            st.session_state.img_to_advise_done = True

    if st.button('다음으로'):
        st.session_state.page = 'result_1'