import streamlit as st
import base64
import image_to_text
from bardapi import Bard
import traceback
from utils import trim_result

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
        안녕, 바드. 너는 세상에서 가장 유능한 영양사이자, 뛰어난 언어학자인 동시에 의사야.

        우리는 입력된 가공식품의 영양정보와, 주의사항에 대한 이해를 바탕으로 고객에게 맞춤 영양정보와 섭취시 주의사항을 제공할거야. 이를 위한 작업은 3개의 프로세스로 구성되어 있어.

        1. 입력 데이터 전처리 
        2. 전처리 데이터에서 내가 제시해준 양식에 맞게 값을 채우기 (주어진 조건에 맞게 계산)
        3. 최종 결과를 제공된 양식에 맞게 작성 후 반환

        아래는 가공식품의 뒷면의 영양성분 구성과 주의사항등이 담긴 글뭉치야. 이를 완벽하게 해석해줄래? 단, 일부 특수문자나 영어가 숫자로 인식되고 있어. %를 8로 읽는다거나, g를 9로 인식하고 있어. 이런 부분들은 수정해서 해석해줘.
        추가로, 숫자 뭉치에서 가장 뒷자리가 '9라면', 예를 들어 429라면, 9를 g으로 바꿔서 해석해줘. 그렇다면 429는 42g으로 해석해야겠지?

        {result_text}

        해석이 완료된 성분표를 의학적으로 분석한 후 아래 [**시작**]과 [**종료**]사이에 있는 양식에 맞게 아래 내용을 채워줘.

        1) 이 음식을 섭취할 때 주의해야 할 알레르기, 특정 질환 환자에 대해 작성해줘.
        2) 개인 칼로리에 따른 영양성분 함유량의 비율을 양식에 맞게 작성해줘.
        3) 이 음식의 칼로리를 모두 소모하기 위해 운동해야하는 시간을 계산해서 작성해줘, 

        이후 반드시 내가 제공해준 양식만 출력해줘. 개인 하루 권장 칼로리는 {st.session_state.user_info['calories']}kcal로 계산해줘.

        [**시작**]
        -----------------------------------------------------------------------------------------------------------------------
        {st.session_state.user_info['user_name']}님, 안녕하세요! {st.session_state.user_info['user_name']}님 맞춤 식품 영양 분석자료입니다. 


        1. {st.session_state.user_info['user_name']}님, 이 가공식품을 섭취할 때 주의해야하는 사항입니다.

        (+) 이런 알레르기가 있으시다면 주의해야 합니다!
        -> 여기에 위험할 것으로 예상되는 알레르기 반응들을 작성해줘.
        (+) 아래 특징을 가지고 있지는 않으신가요?
        -> 여기에 위험할 것으로 예상되는 특정 질환들을 작성해줘.


        2. {st.session_state.user_info['user_name']}님의 1일 기초대사량 대비 영양성분 구성은 다음과 같아요. [이 부분은 표로 작성해줘]

        1) 총 칼로리 : OOkcal (OO%)
        2) 나트륨 : OOmg (OO%)
        3) 탄수화물 : OOg (OO%)
        4) 당류 : OOg (OO%)
        5) 지방 : OOg (OO%)
        6) 트랜스지방 : OOg
        7) 포화지방 : OOg (OO%)
        8) 콜레스테롤 : OOg (OO%)
        9) 단백질 : OOg (OO%)



        3. {st.session_state.user_info['user_name']}님, 이 음식을 소화하기 위해서는 아래와 같은 운동 시간이 필요해요! [표로 작성해줘]

        > 등산 : 위에 분석한 가공식품의 칼로리를 모두 소모하기 위해 등산해야 하는 시간을 계산 (등산은 1분당 10칼로리 소모) : 예를 들어 제품 칼로리가 500칼로리라면 '50분' 반환
        > 러닝 : 위에 분석한 가공식품의 칼로리를 모두 소모하기 위해 러닝해야 하는 시간을 계산 (런닝은 1분당 15칼로리 소모) : 예를 들어 제품 칼로리가 500칼로리라면 '35분' 반환
        > 걷기 : 위에 분석한 가공식품의 칼로리를 모두 소모하기 위해 걷기를 해야하는 시간을 계산 (걷기는 1분당 7칼로리 소모) : 예를 들어 제품 칼로리가 500칼로리라면 '70분' 반환
        > 수영 : 위에 분석한 가공식품의 칼로리를 모두 소모하기 위해 걷기를 해야하는 시간을 계산 (수영는 1분당 8칼로리 소모) : 예를 들어 제품 칼로리가 500칼로리라면 '63분' 반환
        > 필라테스 : 위에 분석한 가공식품의 칼로리를 모두 소모하기 위해 걷기를 해야하는 시간을 계산 (필라테스는 1분당 6칼로리 소모) : 예를 들어 제품 칼로리가 500칼로리라면 '85분' 반환
        > 농구 : 위에 분석한 가공식품의 칼로리를 모두 소모하기 위해 걷기를 해야하는 시간을 계산 (농구는 1분당 12칼로리 소모) : 예를 들어 제품 칼로리가 500칼로리라면 '42분' 반환
        ------------------------------------------------------------------------------------------------------------------------------
        
        [**종료**]

        최종 작성된 양식의 형태에서 벗어나는 부가적인 내용은 모두 제거한 후 최종적으로 [**시작**]과 [**종료**] 사이의 값을 반환해줘.

        어린아이와 젊은 층이 읽었을 때 다소 딱딱하게 느껴지지 않도록 부드러운 말투와 이모티콘을 섞어서 작성해주고, 가시성이 좋게 문장간 공백을 포함해서 작성해줘.
        """
        
        # bard api 호출
        if 'api_response' not in st.session_state:
            st.session_state.api_response = list()
        try:
            print('bard api 호출 시도')
            original_result = Bard().get_answer(str(query))['content']
            print("====================original_api_response========================")
            print(original_result)
            print("=============================================================================")
            trimmed_result = trim_result(original_result)
            st.session_state.api_response.append(trimmed_result)
            print("====================st.session_state.api_response[-1]========================")
            print(st.session_state.api_response[-1])
            print("=============================================================================")
        except Exception as e:
            print(f"bard api 호출 실패")
            traceback.print_exc()
            st.warning('bard api 호출 실패!')
            exit()
        else:
            st.session_state.img_to_advise_done = True

    if st.button('다음으로'):
        st.session_state.page = 'result_1'

