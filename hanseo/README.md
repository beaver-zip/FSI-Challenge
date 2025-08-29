# FSI-Challenge: 금융보안 질의응답 RAG 시스템

이 프로젝트는 금융보안 관련 질문에 답변하기 위한 Retrieval-Augmented Generation (RAG) 시스템을 구현합니다. 질문의 주제(법률, 금융, 보안)에 따라 적절한 답변 전략(RAG 또는 Zero-shot)과 모델(기본 LLM 또는 LoRA 튜닝 LLM)을 사용하여 정확하고 효율적인 답변을 제공합니다.

현재 버전:
- 법률 : RAG, 기본모델
- 금융/보안 : Zero-Shot, LoRA

## 주요 기능

*   **PDF 문서 처리**: `pdfs/` 디렉토리에 있는 PDF 문서에서 텍스트를 추출하고, 청크로 분할하여 RAG 시스템에 활용합니다.
*   **로컬 벡터 데이터베이스**: `upskyy/bge-m3-korean` 임베딩 모델과 FAISS를 사용하여 문서 청크의 임베딩을 생성하고 효율적인 검색을 위한 벡터 데이터베이스를 구축합니다.
*   **질문 주제 분류**: 입력된 질문을 "법률", "금융", "보안" 세 가지 주제 중 하나로 분류하여 최적의 답변 전략을 결정합니다.
*   **조건부 RAG/생성**:
    *   **법률 관련 질문**: PDF 문서에서 검색된 관련 정보를 기반으로 RAG를 수행하여 답변을 생성합니다.
    *   **금융/보안 관련 질문**: LoRA 튜닝된 LLM을 사용하여 Zero-shot 방식으로 답변을 생성합니다.
*   **대규모 언어 모델 활용**: `K-intelligence/Midm-2.0-Base-Instruct` 모델을 기본 LLM으로 사용하며, 특정 주제에 대한 성능 향상을 위해 LoRA 어댑터를 적용합니다.

## 프로젝트 구조

```
FSI-Challenge/
├── hanseo/
│   ├── baseline.ipynb             # 베이스라인 모델 개발을 위한 Jupyter 노트북
│   ├── README.md                  # 프로젝트 설명 (현재 파일)
│   ├── requirements.txt           # Python 종속성 목록
│   ├── ckpt_tapt_mcqa_stage1/     # LoRA 어댑터 및 토크나이저 관련 파일
│   │   └── lora_adapter/          # LoRA 어댑터 가중치
│   ├── model/                     # 기본 LLM 모델 파일
│   └── RAG/
│       ├── doc_metadata.pkl       # 문서 메타데이터 (벡터 DB용)
│       ├── faiss_index.bin        # FAISS 벡터 인덱스 파일
│       ├── run.py                 # 메인 실행 스크립트 (RAG 시스템)
│       ├── sample_submission.csv  # 제출 양식 예시
│       ├── submission.csv         # 최종 제출 파일
│       ├── test.csv               # 테스트 질문 데이터
│       ├── pdfs/                  # RAG에 사용될 PDF 문서들
│       │   └── *.pdf
│       ├── Preprocessor/
│       │   └── preprocessor.py    # PDF 문서 전처리 스크립트
│       └── VDB/
│           └── localVDB.py        # 로컬 벡터 데이터베이스 관리 스크립트
└── data_process/                  # 데이터 처리 관련 스크립트 (추정)
```

## 설치 및 실행 방법

### 1. 환경 설정

Python 3.8 이상 버전이 필요합니다. 가상 환경을 사용하는 것을 권장합니다.

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### 2. 종속성 설치

`requirements.txt` 파일에 명시된 모든 라이브러리를 설치합니다.

```bash
pip install -r hanseo/requirements.txt
```

### 3. PDF 문서 준비

RAG 시스템에 사용할 PDF 문서들을 `FSI-Challenge/hanseo/RAG/pdfs/` 디렉토리에 넣어주세요.

### 4. 모델 및 인덱스 생성/다운로드

`run.py` 스크립트를 처음 실행하면, 필요한 LLM 모델(`K-intelligence/Midm-2.0-Base-Instruct`)과 임베딩 모델(`upskyy/bge-m3-korean`)이 자동으로 다운로드됩니다. 또한, `pdfs/` 디렉토리의 문서들을 기반으로 FAISS 인덱스와 문서 메타데이터가 생성됩니다.

### 5. 시스템 실행

테스트 질문(`test.csv`)에 대한 답변을 생성하고 `submission.csv` 파일을 만듭니다.

```bash
python hanseo/RAG/run.py
```

실행이 완료되면 `FSI-Challenge/hanseo/RAG/submission.csv` 파일에 결과가 저장됩니다.

## 사용된 기술 스택

*   **Python**
*   **PyTorch**
*   **Hugging Face Transformers**: LLM 로드 및 추론
*   **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA 어댑터 적용
*   **Sentence-Transformers**: 문서 임베딩 생성
*   **FAISS**: 벡터 검색
*   **PyMuPDF (fitz)**: PDF 문서 처리
*   **LangChain**: 텍스트 분할 (RecursiveCharacterTextSplitter)
*   **Pandas**: 데이터 처리
*   **tqdm**: 진행률 표시

## 기여자

[당신의 이름 또는 팀 이름]

## 라이선스

[프로젝트 라이선스 정보]