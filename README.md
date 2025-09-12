# 금융보안원 AI 챌린지 코드 리팩토링 프로젝트

## 1. 프로젝트 구조

```
/
├── streamlit_app.py       # Streamlit 실행 앱
├── inference.py           # 추론 함수
├── requirements.txt       # Python 의존성 목록
├── README.md              # 실행 가이드
├── .env                   # Hugging Face 토큰
│
├── src/                   # 공용 모듈
│   ├── rag.py             # RAG 관련 클래스/함수
│   └── preprocessor.py    # 문서 전처리 클래스
│
├── data/
│   ├── raw/               # 원본 데이터
│   │   ├── adapter_docs/  # .doc 법률 문서 (DAPT용)
│   │   ├── RAG_pdfs/      # .pdf 법률 문서 (RAG용)
│   │   └── mcqa_dataset/  # aiqwe/FinShibainu 데이터셋
│   ├── processed/         # 전처리된 데이터 (dapt_corpus.txt 등)
│   └── external_data_proof.md # 외부 데이터 출처 증빙 (필요시 작성)
│
├── scripts/
│   ├── 0_download_assets.py # 모델/데이터셋 다운로드 스크립트
│   ├── 1_prepare_data.py    # 법령 데이터 전처리 스크립트
│   ├── 2_run_dapt.py        # DAPT 학습 스크립트
│   ├── 3_run_tapt.py        # TAPT 학습 스크립트
│   └── 4_build_vectordb.py  # Faiss 벡터 DB 생성 스크립트
│
├── models/
│   ├── base_model/          # 사전 다운로드한 베이스 LLM
│   └── tapt_adapter/        # 최종 LoRA 어댑터
│
└── db/
    ├── faiss_index.bin      # FAISS 인덱스
    └── doc_metadata.pkl     # 문서 메타데이터
```

## 2. 실행 방법

### 2.1. 환경 설정

1.  **가상 환경 생성 및 활성화**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **필수 라이브러리 설치**
    ```bash
    pip install -r requirement.txt
    ```

3.  **Hugging Face Hub 토큰 설정**
    - 프로젝트 루트에 있는 `.env` 파일을 엽니다.
    - `HUGGING_FACE_HUB_TOKEN=''`의 따옴표 안에 본인의 Hugging Face Access Token을 입력하고 저장합니다. (예: `HUGGING_FACE_HUB_TOKEN='hf_xxxxxxxx'`)

### 2.2. 전체 파이프라인 실행

1.  **[최초 1회] 모델 및 데이터셋 다운로드**
    - `K-intelligence/Midm-2.0-Base-Instruct` 모델과 `aiqwe/FinShibainu` 데이터셋을 로컬에 다운로드합니다.
    ```bash
    python scripts/0_download_assets.py
    ```

2.  **법령 데이터 전처리**
    - `data/raw/adapter_docs` 안의 문서들을 DAPT 학습을 위한 `dapt_corpus.txt` 파일로 변환합니다.
    ```bash
    python scripts/1_prepare_data.py
    ```

3.  **DAPT(Domain-Adaptive Pre-training) 실행**
    - 전처리된 코퍼스로 도메인 적응 학습을 진행하고 `models/dapt_adapter`에 LoRA 가중치를 저장합니다.
    ```bash
    python scripts/2_run_dapt.py
    ```

4.  **TAPT(Task-Adaptive Pre-training) 실행**
    - DAPT 어댑터를 이어받아 MCQA 데이터셋으로 추가 학습을 진행하고, 최종 LoRA 가중치를 `models/tapt_adapter`에 저장합니다.
    ```bash
    python scripts/3_run_tapt.py
    ```

5.  **벡터 데이터베이스 생성**
    - `data/raw/RAG_pdfs`의 PDF 문서들을 임베딩하여 `db/` 폴더에 Faiss 인덱스를 생성합니다.
    ```bash
    python scripts/4_build_vectordb.py
    ```

### 2.3. 추론 앱 실행 (사전 빌드된 모델/DB 사용 시)

- 위의 전체 파이프라인을 통해 `models/tapt_adapter`와 `db/` 폴더가 준비되었다고 가정합니다.
- `streamlit_app.py`가 `inference.py`를 호출하여 웹 기반 질의응답 앱을 실행합니다.

```bash
streamlit run streamlit_app.py
```
