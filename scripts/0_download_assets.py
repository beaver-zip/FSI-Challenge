# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# .env 파일에서 환경 변수를 로드합니다.
# 이 함수는 HUGGING_FACE_HUB_TOKEN 같은 변수를 os.environ에 설정합니다.
load_dotenv()
from datasets import load_dataset

# 이 스크립트의 위치를 기준으로 경로 설정
CWD = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CWD, '..'))

# --- 다운로드할 에셋 정보 ---
LLM_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
MCQA_DATASET_ID = "aiqwe/FinShibainu"
MCQA_DATASET_NAME = "mcqa"

# --- 저장 경로 ---
LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "base_model")
MCQA_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "mcqa_dataset")

def main():
    """
    추론 및 학습에 필요한 모델과 데이터셋을 로컬에 다운로드합니다.
    """
    print(f"--- 1. LLM 베이스 모델 다운로드 시작 ---")
    print(f"모델 ID: {LLM_MODEL_ID}")
    print(f"저장 경로: {LLM_MODEL_PATH}")
    
    if os.path.exists(LLM_MODEL_PATH) and os.listdir(LLM_MODEL_PATH):
        print("모델이 이미 존재합니다. 다운로드를 건너뜁니다.")
    else:
        os.makedirs(LLM_MODEL_PATH, exist_ok=True)
        snapshot_download(
            repo_id=LLM_MODEL_ID,
            local_dir=LLM_MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("모델 다운로드 완료.")

    print(f"\n--- 2. MCQA 데이터셋 다운로드 시작 ---")
    print(f"데이터셋 ID: {MCQA_DATASET_ID}")
    print(f"저장 경로: {MCQA_DATASET_PATH}")

    if os.path.exists(MCQA_DATASET_PATH) and os.listdir(MCQA_DATASET_PATH):
        print("데이터셋이 이미 존재합니다. 다운로드를 건너뜁니다.")
    else:
        os.makedirs(MCQA_DATASET_PATH, exist_ok=True)
        load_dataset(
            MCQA_DATASET_ID, 
            name=MCQA_DATASET_NAME,
            cache_dir=MCQA_DATASET_PATH # cache_dir를 저장 경로로 사용하여 파일을 해당 위치에 저장
        )
        print("데이터셋 다운로드 완료.")
        # Hugging Face datasets는 cache_dir 내부에 복잡한 구조로 저장하므로,
        # 사용자가 직접 파일을 옮길 필요 없이 이 경로를 학습 스크립트에서 사용하도록 안내해야 합니다.

if __name__ == "__main__":
    main()
