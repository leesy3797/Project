import streamlit as st
import pandas as pd
import sqlite3
import json
import datetime
import sys
import subprocess

# Set page configuration
st.set_page_config(layout="centered")
st.title("정부 지원 사업 Check Agent")
col1, col2, col3, col4 = st.columns(4)

for i in range(1, 5):
    if not st.session_state.get(f'result{i}'):
        st.session_state[f'result{i}'] = None

def run_script(script_path, button_label):
    with st.spinner("실행 중..."):
        try:
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            return result
        except Exception as e:
            return 'error'  # 에러 발생 시 None 반환

with col1:
    if st.button('**[기업마당] 수집**'):
        st.session_state['result1'] = run_script('crawler/crawl_bizinfo_final.py', "기업마당 수집")

with col2:
    if st.button('**[중기부] 수집**'):
        st.session_state['result2'] = run_script('crawler/crawl_mss_final.py', "중기부 수집")

with col3:
    if st.button('**LongList 필터링**'):
        st.session_state['result3'] = run_script('crawler/filtering_long_list.py', "Long List 필터링")

with col4:
    if st.button('**ShortList 필터링**'):
        st.session_state['result4'] = run_script('crawler/filtering_short_list.py', "Short List 필터링")

def display_result(result, error_message):
    if result:
        # 파란색, 볼드체로 표시
        if f'%%' in result.stdout: 
            st.markdown(f"<p style='color:blue; font-weight:bold;'>{result.stdout.split(f'%%')[-2]}</p>", unsafe_allow_html=True)
        elif f'%%' in result.stderr:
            st.markdown(f"<p style='color:blue; font-weight:bold;'>{result.stderr.split(f'%%')[-2]}</p>", unsafe_allow_html=True)
            # st.markdown(f"<p style='color:blue; font-weight:bold;'>{result}</p>", unsafe_allow_html=True)
        else:
            # 빨간색, 볼드체로 표시
            st.markdown(f"<p style='color:red; font-weight:bold;'>{error_message}</p>", unsafe_allow_html=True)
            with st.expander('**⚠️ Error Log**'):
                st.markdown(f'{result.stderr}')
    else:
        # 초록색, 볼드체로 표시
        st.markdown(f"<p style='color:green; font-weight:bold;'>[대기 항목] {error_message.split(' 중 ')[0]}</p>", unsafe_allow_html=True)
  

# 함수 사용 예제
display_result(st.session_state['result1'], "기업마당 수집 중 에러 발생")
display_result(st.session_state['result2'], "중기부 수집 중 에러 발생")
display_result(st.session_state['result3'], "Long List 필터링 중 에러 발생")
display_result(st.session_state['result4'], "Short List 필터링 중 에러 발생")

st.divider()

# Connect to SQLite database and fetch data
try:
    conn = sqlite3.connect("crawler/announcement_sqlite.db")
    cursor = conn.cursor()
    cursor.execute("select * from announcements;")
    data = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    cursor.close()
    conn.close()
    announcement_df = pd.DataFrame(data=data, columns=columns)
except:
    conn.close()
    announcement_df = pd.DataFrame()

# Load the JSONL file
with open("crawler/downloads/short_list.jsonl", mode='r', encoding='utf-8') as files:
    # contents = []
    # for line in files:
    #     try:
    #         contents.append(json.loads(line.strip()))
    #     except json.JSONDecodeError:
    #         print(f"Invalid JSON format in line: {line}")
    contents = [json.loads(line) for line in files]
shortlist_df = pd.DataFrame(contents)

# Merge DataFrames
merge_df = pd.merge(shortlist_df, announcement_df, how='left', left_on='사업 공고명', right_on='title')

# Ensure 'Planning' column exists
if 'Planning' not in merge_df.columns:
    merge_df['Planning'] = False 

# --- Part 1: KPI Metrics ---
# Toggle button for status (Recommended or All)
st.subheader("🌐:blue[정부 과제 목록]")
show_recommended_only = st.toggle("**추천 공고만 보기**", value=False)

# Apply filters based on toggle status (for Project List, not for Metrics)
if show_recommended_only:
    filtered_df = merge_df[merge_df['결과'] == '추천']
else:
    filtered_df = merge_df  # Show all if toggle is off

# Update 'Planning' count based on the edited dataframe
# First display the 'data_editor' to allow changes to the 'Planning' column

with st.expander(":mag:**Short List 목록**"):
    edited_df = st.data_editor(
        filtered_df[['Planning', '결과', '사업 공고명', 'application_period', 'attachment', 'url', 'posted_date', 'department', '사업 형식', '사업 개요(요약)']],
        column_config={
            'Planning': st.column_config.CheckboxColumn("준비중"),
            '결과': st.column_config.TextColumn("추천여부"),
            'application_period': st.column_config.TextColumn("신청기간"),
            'attachment': st.column_config.LinkColumn("첨부파일", display_text="file"),
            'url': st.column_config.LinkColumn("URL", display_text="link"),
            'posted_date': st.column_config.TextColumn("공고일")
        },
        hide_index=True
    )

# Now that the editor has been displayed, update the session state with the edited values
st.session_state['Planning'] = edited_df['Planning']

# Update planning count after the user has edited the data
planning_count = edited_df[edited_df['Planning'] == True].shape[0]

# Metrics calculation
total_announcement = merge_df.shape[0]
recommended_count = merge_df[merge_df['결과'] == '추천'].shape[0]
recommended_percentage = (recommended_count / total_announcement) * 100 if total_announcement > 0 else 0

tmp_df = merge_df[pd.to_datetime(merge_df["posted_date"]) <= pd.to_datetime(datetime.date.today() - datetime.timedelta(days=7))]
total_announcement_prev = tmp_df.shape[0]
recommended_count_prev = tmp_df[tmp_df['결과'] == '추천'].shape[0]
recommended_percentage_prev = (recommended_count_prev / total_announcement_prev) * 100 if total_announcement_prev > 0 else 0

total_announcement_delta = total_announcement - total_announcement_prev
recommended_delta = recommended_count - recommended_count_prev

st.divider()
# Display metrics (Summary Metrics should not be affected by toggle)
st.subheader("📈:blue[주요 지표]")
col1, col2, col3 = st.columns(3)

col1.metric("**총 공고수**", total_announcement, f"{total_announcement_delta} from last week")
col2.metric("**추천 공고수**", f"{recommended_count} ({recommended_percentage:.2f}%)", f"{recommended_delta} ({(recommended_delta / total_announcement) * 100:.2f}%) from last week")
col3.metric("**계획중인 공고**", planning_count)

st.divider()

# Part 3: Calendar View with Project Deadlines
st.subheader("✅:blue[공고 신청 기간 체크]")

# Filter for projects marked as 'Planning'
planning_df = edited_df[edited_df['Planning'] == True]

# Calendar function to visualize deadlines
def show_calendar(df):
    if df.empty:
        # st.write("No planned projects available. Here's an empty calendar.")
        # st.dataframe(merge_df)
        pass
    else:
        with st.container():
            for index, row in df.iterrows():
                try:
                    start_date, end_date = row['application_period'].split(" ~ ")
                    st.markdown(f"""
                        <div style='background-color:#f0f8ff; padding:10px; border-radius:5px; margin-bottom:10px;'>
                            <strong>ㅁ 과제명 :</strong> {row['사업 공고명']} <br>
                            <strong> o 주관 부서 :</strong> {row['department']} <br>
                            <strong> o 지원기간 :</strong> {start_date} ~ {end_date} <br>
                            <strong> o 지원 구분 :</strong> {row['사업 형식']} <br>
                            <strong> o 사업 개요 :</strong> {row['사업 개요(요약)']}  
                        </div>
                    """, unsafe_allow_html=True)
                except:
                    st.markdown(f"""
                        <div style='background-color:#f0f8ff; padding:10px; border-radius:5px; margin-bottom:10px;'>
                            <strong>ㅁ 과제명 :</strong> {row['사업 공고명']} <br>
                            <strong> o 주관 부서 :</strong> {row['department']} <br>
                            <strong> o 지원기간 :</strong> {row['application_period']} <br>
                            <strong> o 지원 구분 :</strong> {row['사업 형식']} <br>
                            <strong> o 사업 개요 :</strong> {row['사업 개요(요약)']}
                        </div>
                    """, unsafe_allow_html=True)


# Display calendar for 'Planning' projects
show_calendar(planning_df)

# Footer
st.caption("Dashboard built by Lee Seung Yong")