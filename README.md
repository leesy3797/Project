# 프로젝트 포트폴리오 🚀

## 1. [OCR과 LLM을 활용한 개인 맞춤 영양정보 제공 End-to-End 서비스 🥗]([프로젝트]%20OCR과%20LLM을%20활용한%20개인%20맞춤%20영양정보%20제공%20End-to-End%20서비스/README.md)
- **기간**: 2023.10.08 ~ 2023.10.26
- **주제**: 이미지로 입력받은 가공식품의 영양 정보를 분석하여 개인 맞춤형 건강 정보를 제공하는 End to End 서비스 개발
- **주요 성과**:
  - OCR 모델 성능 비교 및 이미지 전처리 방법 연구 (Tesseract OCR, EasyOCR, Paddle OCR)
  - EasyOCR Fine tuning을 통한 Text Recognition 성능 향상 시도
  - Streamlit 기반 Web 서비스 프로토타입 구현
  - Bard API를 활용한 개인 맞춤형 건강 정보 제공 기능 개발
- **활용 기술**: Python, OpenCV, Tesseract OCR, EasyOCR, Paddle OCR, Bard API, Streamlit

## 2. [국책과제 AI Assistant 🤖]([PoC]%20국책과제%20AI%20Assistant/README.md)
- **기간**: 2024.08 ~ 2024.09 (1.5개월)
- **주제**: LLM과 RAG 시스템을 활용한 국책과제 공고문 검토 시간 단축 및 업무 프로세스 개선
- **주요 성과**:
  - 사업 공고문 검토 시간 단축: 주 평균 6시간 → 1시간 (약 83% 감소)
  - RAG 시스템 3단계 성능 개선 (v1.0 → v3.0)
  - LLM 출력 결과의 신뢰성 향상 (Recall 성능 20% 개선)
  - 한글 문서 처리 시스템 자체 개발
- **활용 기술**: Python, Gemini, Langchain, Faiss, ElasticSearchBM25, SQLite, Streamlit, Prefect

## 3. [통신장비 예지보전 및 기지국 관리를 위한 이상탐지 시스템 📡]([프로젝트]%20통신장비%20예지보전%20및%20기지국%20관리를%20위한%20이상탐지%20시스템/README.md)
- **기간**: 2023.11.07 ~ 2024.01.07
- **주제**: 시계열 통신 데이터를 기반으로 다양한 접근법을 활용하여 이상 시그널을 탐지하고 효율적인 유지 보수를 지원하는 AI 모델 및 대시보드 개발
- **주요 성과**:
  - F0.5 기준 0.6224의 성능 달성 (목표 대비 4.7% 향상)
  - 3가지 이상탐지 방법론(Z-Score, Classification, Isolation Forest) 앙상블을 통한 신뢰성 높은 이상탐지 파이프라인 구축
  - 73,440개 평가 데이터 중 8,812개(12%) 이상치 검출
  - 출동 비용 30% 절감 및 탐지 신뢰도 개선 (False Alarm 감소)
  - 지역별/일자별 지표, 이상 기지국 추적, 이상 요인 분석 등 직관적인 모니터링 시스템 구축
- **활용 기술**: Python, Scikit-learn, Matplotlib, Seaborn, Pycaret, Tableau

## 4. [신규 외식 창업자들을 위한 서울시 외식 창업 시장 분석 🍽️]([프로젝트]%20신규%20외식%20창업자들을%20위한%20서울시%20외식%20창업%20시장%20분석/README.md)
- **기간**: 2023.04 ~ 2023.06
- **주제**: 서울시 외식업 시장 데이터를 분석하여 신규 창업자들을 위한 시장 현황 및 전략 수립
- **주요 성과**:
  - 소매포화지수(IRS) 기반 시장 경쟁도 평가 체계 구축
  - 시장 안정성 지표 = (개업률 × 3년 생존율) / 폐업률 개발
  - 서울시 25개 구의 외식업 시장 현황 분석 및 업종별, 지역구별 시장 포지셔닝 맵 작성
  - Tableau를 활용한 인터랙티브 대시보드 개발
- **활용 기술**: Python, BeautifulSoup, Selenium, Pandas, NumPy, Matplotlib, Seaborn, Tableau 