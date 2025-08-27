# 금융보안원 AI 챌린지 2024 - RAG 기반 질의응답 시스템

## 1. 프로젝트 개요

본 프로젝트는 금융보안 분야의 질문에 대해 정확하고 신뢰성 있는 답변을 제공하는 QA(Question Answering) 시스템입니다. 사용자의 질문 유형을 LLM을 통해 동적으로 파악하고, 질문이 법률과 관련될 경우에만 RAG(Retrieval-Augmented Generation) 기술을 적용하여 답변의 정확도를 높이는 **조건부 RAG (Conditional RAG)** 방식을 핵심 전략으로 사용합니다.

- **LLM**: `K-intelligence/Midm-2.0-Base-Instruct`
- **Embedding Model**: `upskyy/bge-m3-korean`
- **Vector DB**: `FAISS` (Facebook AI Similarity Search)

---

## 2. 프로젝트 구조

```
FSI-Challenge/
└── hanseo/
    ├── model/                     # LLM 모델 저장 디렉토리 (자동 생성)
    ├── RAG/
    │   ├── Preprocessor/
    │   │   └── preprocessor.py    # PDF 문서 전처리기
    │   ├── VDB/
    │   │   └── localVDB.py        # FAISS 벡터 DB 관리 모듈
    │   ├── pdfs/                  # 법률 정보 PDF 문서
    │   │   ├── credit_information_law.pdf
    │   │   ├── digital_sign_law.pdf
    │   │   ├── network_law.pdf
    │   │   ├── privacy_law.pdf
    │   │   └── transaction_law.pdf
    │   ├── doc_metadata.pkl       # 문서 메타데이터 (자동 생성)
    │   ├── faiss_index.bin        # FAISS 인덱스 파일 (자동 생성)
    │   ├── run.py                 # 메인 실행 스크립트
    │   ├── sample_submission.csv  # 제출 양식 샘플
    │   └── test.csv               # 테스트 질문 데이터
    ├── README.md                  # 프로젝트 설명서
    └── requirements.txt           # 필요 패키지 목록
```

---

## 3. 전체적인 동작 원리

본 시스템은 `run.py` 스크립트 하나로 모든 과정이 실행됩니다. 동작 순서는 다음과 같습니다.

1.  **초기 설정**:
    - `run.py` 실행 시, 먼저 `hanseo/model` 폴더에 LLM이 존재하는지 확인합니다. 만약 모델이 없으면 Hugging Face Hub에서 자동으로 다운로드합니다.
    - `RAG/` 폴더 내에 FAISS 벡터 인덱스(`faiss_index.bin`)가 있는지 확인합니다.

2.  **벡터 DB 생성 (최초 실행 시)**:
    - FAISS 인덱스가 없다면, `pdfs/` 폴더의 법률 문서들을 `preprocessor.py`를 이용해 의미 있는 단위(chunk)로 분할합니다.
    - `localVDB.py`가 분할된 텍스트 조각들을 임베딩 모델(`bge-m3-korean`)을 사용해 벡터로 변환합니다.
    - 변환된 벡터들을 `FAISS` 인덱스로 구축하고, `faiss_index.bin` 파일로 저장하여 다음 실행부터는 이 과정을 생략합니다.

3.  **질의응답 추론**:
    - `test.csv`에서 질문을 순서대로 읽어옵니다.
    - **(핵심 로직)** 각 질문을 LLM에 보내 주제를 "법률", "금융", "보안" 중 하나로 분류합니다.
    - **조건부 RAG 적용**:
        - **"법률"** 질문: `localVDB.py`를 통해 FAISS DB에서 질문과 가장 유사한 법률 문서 조각 3개를 검색합니다. 검색된 내용을 참고자료로 삼아 프롬프트를 구성하고, LLM에 답변 생성을 요청합니다.
        - **"금융" 또는 "보안" 질문**: 별도의 외부 자료 없이, LLM이 가진 자체 지식만으로 답변을 생성하는 Zero-shot 방식으로 프롬프트를 구성하여 질의합니다.
    - 생성된 답변은 후처리 함수를 통해 문제 유형(객관식/주관식)에 맞게 정제됩니다.

4.  **제출 파일 생성**:
    - 모든 질문에 대한 답변 생성이 완료되면, `sample_submission.csv` 형식에 맞춰 `submission.csv` 파일을 생성합니다.

---

## 4. 각 파일 및 폴더 설명

- **`run.py`**:
    - 전체 프로세스를 총괄하는 메인 스크립트입니다.
    - 모델 및 벡터 DB 로드, 질문 분류, 조건부 RAG, 추론, 제출 파일 생성 등 모든 작업을 순차적으로 실행합니다.
    - 최초 실행 시 필요한 모델과 벡터 DB를 자동으로 생성하는 기능이 포함되어 있습니다.

- **`RAG/Preprocessor/preprocessor.py`**:
    - PDF 문서를 텍스트로 변환하고, LLM이 처리하기 용이하도록 의미 있는 문단 단위로 분할하는 역할을 합니다.

- **`RAG/VDB/localVDB.py`**:
    - 임베딩 모델을 로드하고 FAISS 벡터 데이터베이스를 관리하는 클래스입니다.
    - 텍스트를 벡터로 변환, FAISS 인덱스 생성/저장/로드, 유사도 검색 기능을 제공합니다.

- **`RAG/pdfs/`**:
    - RAG의 기반 지식이 되는 법률 관련 PDF 문서들이 저장된 폴더입니다.

- **`requirements.txt`**:
    - 프로젝트 실행에 필요한 파이썬 라이브러리 목록입니다. `pip install -r requirements.txt` 명령어로 설치할 수 있습니다.

- **`model/`**:
    - Hugging Face Hub에서 다운로드한 LLM 모델 파일들이 저장되는 폴더입니다. `run.py` 최초 실행 시 자동으로 생성됩니다.

---

## 5. 실행 방법

1.  **필요 패키지 설치**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **스크립트 실행**:
    `RAG` 폴더로 이동하여 `run.py`를 실행합니다.
    ```bash
    cd hanseo/RAG
    python run.py
    ```
    - 최초 실행 시 LLM 모델과 FAISS 인덱스를 다운로드 및 생성하므로 시간이 다소 소요될 수 있습니다.
    - 실행이 완료되면 `RAG` 폴더 내에 `submission.csv` 파일이 생성됩니다.