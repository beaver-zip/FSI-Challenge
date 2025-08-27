import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download

# --- 경로 설정 (run.py 기반) ---
LLM_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "model"))
TEST_CSV_PATH = os.path.join(SCRIPT_DIR, "test.csv")
OUTPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "divide_test.csv")

# --- 도메인 분류 함수 (run.py와 동일) ---
def classify_question_with_llm(question: str, pipe) -> str:
    """
    LLM을 사용하여 질문의 주제를 "법률", "금융", "보안" 중 하나로 분류합니다.
    """
    classification_prompt = f'''다음 질문의 주제를 "법률", "금융", "보안" 중 하나로만 답하세요.

질문: 전자금융거래법 제10조에 명시된 이용자의 권리가 아닌 것은?
주제: 법률

질문: 피싱 공격을 예방하기 위한 사용자 측면의 보안 수칙으로 옳지 않은 것은?
주제: 보안

질문: 변동성이 큰 시장에서 안정적인 수익을 추구하는 투자 포트폴리오 구성 방법은?
주제: 금융

질문: {question}
주제:'''

    raw_output = pipe(classification_prompt, max_new_tokens=5, do_sample=False)

    if "주제:" in raw_output[0]['generated_text']:
        answer = raw_output[0]['generated_text'].split("주제:")[-1].strip()
    else:
        answer = raw_output[0]['generated_text'].strip()

    if "보안" in answer:
        return "보안"
    elif "금융" in answer:
        return "금융"
    else:
        return "법률" # 기본값: 법률

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    # 1. LLM 모델 다운로드 및 로드
    if not os.path.exists(LLM_MODEL_PATH):
        print(f"LLM 모델({LLM_MODEL_ID})을 다운로드합니다. 경로: {LLM_MODEL_PATH}")
        snapshot_download(
            repo_id=LLM_MODEL_ID,
            local_dir=LLM_MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("LLM 모델 다운로드 완료.")
    else:
        print(f"기존 LLM 모델을 사용합니다. 경로: {LLM_MODEL_PATH}")

    print("메인 LLM을 로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    print("LLM 로드 완료.")

    # 2. 테스트 데이터 로드
    print(f"테스트 데이터를 로드합니다: {TEST_CSV_PATH}")
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
    except FileNotFoundError:
        print(f"오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")
        exit()

    # 3. 도메인 분류 실행
    results = []
    print("도메인 분류를 시작합니다...")
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="도메인 분류 중"):
        question_id = row['ID']
        question_text = row['Question']

        domain = classify_question_with_llm(question_text, pipe)
        results.append({'ID': question_id, 'Domain': domain})

    print("도메인 분류 완료.")

    # 4. 결과 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"분류 결과를 '{OUTPUT_CSV_PATH}' 파일로 성공적으로 저장하였습니다.")
