import os
import re
import glob
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download

from Preprocessor.preprocessor import DocumentProcessor
from VDB.localVDB import LocalVectorDB
from peft import PeftModel

# --- 경로 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV_PATH = os.path.join(SCRIPT_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(SCRIPT_DIR, "sample_submission.csv")
SUBMISSION_CSV_PATH = os.path.join(SCRIPT_DIR, "submission.csv")

PDF_PATH = os.path.join(SCRIPT_DIR, "pdfs")
LLM_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
LLM_MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "model"))
LORA_ASSETS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ckpt_tapt_mcqa_stage1")) # For Tokenizer
LORA_ADAPTER_PATH = os.path.join(LORA_ASSETS_PATH, "lora_adapter") # For Adapter weights
EMBEDDING_MODEL_NAME = "upskyy/bge-m3-korean"

FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_index.bin")
DOC_META_PATH = os.path.join(SCRIPT_DIR, "doc_metadata.pkl")

# 객관식 여부 판단 함수 (baseline)
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

# 질문과 선택지 분리 함수 (baseline)
def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())

    question = " ".join(q_lines)
    return question, options

# 후처리 함수 (baseline)
def extract_answer_only(generated_text: str, original_question: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식 문제면: 정답 숫자만 추출 (실패 시 전체 텍스트 또는 기본값 반환)
    - 주관식 문제면: 전체 텍스트 그대로 반환
    - 공백 또는 빈 응답 방지: 최소 "미응답" 반환
    """
    # "답변:" 기준으로 텍스트 분리
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()

    # 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    if is_multiple_choice(original_question):
        # 숫자만 추출
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            return "0" # 숫자 추출 실패 시 "0" 반환
    else:
        return text

def make_rag_prompt(question_text, context):
    question, options = extract_question_and_choices(question_text)
    context_str = "\n\n".join(context)

    if is_multiple_choice(question_text): # RAG 객관식 프롬프트
        return (
            "당신은 금융보안 전문가입니다. 주어진 [참고 자료]와 당신의 지식을 활용하여, 다음 [질문]에 가장 적절한 **선택지 번호 하나만** 고르세요.\n"
            "**반드시 주어진 선택지 중 하나의 번호를 답변해야 합니다.**\n\n"
            "[참고 자료]\n"
            f"{context_str}\n\n"
            "[질문]\n"
            f"{question}\n\n"
            "[선택지]\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else: # RAG 주관식 프롬프트
        return (
            "당신은 금융보안 전문가입니다. 다음 질문에 대해 당신의 전문 지식을 활용하여 답변하세요.\n"
            "답변을 작성할 때, 주어진 [참고 자료]를 최대한 활용하되, 자료에 내용이 없거나 부족하더라도 반드시 질문에 대한 답변을 해야 합니다.\n"
            "**'자료에 내용이 없다'거나 '알 수 없다'는 식의 답변은 절대 하지 마세요.**\n"
            "답변은 **네 문장 이내로 정확하고 간결하게** 서술하세요.\n\n"
            "[질문]\n"
            f"{question_text}\n\n"
            "[참고 자료]\n"
            f"{context_str}\n\n"
            "답변:"
        )

# baseline.ipynb의 프롬프트 생성기
def make_baseline_prompt(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
                )
    else:
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 주관식 질문에 대해 **네 문장 이내로 정확하고 간결하게** 답변하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
                )
    return prompt




# LLM 호출 1: 주제 분류 (법률은 pdfs에 있는 문서로 RAG, 보안/금융은 Zero-shot으로 처리)
def classify_question_with_llm(question: str, pipe) -> str:
    classification_prompt = f"""다음 질문의 주제를 "법률", "금융", "보안" 중 하나로만 답하세요.

질문: 전자금융거래법 제10조에 명시된 이용자의 권리가 아닌 것은?
주제: 법률
    
질문: 피싱 공격을 예방하기 위한 사용자 측면의 보안 수칙으로 옳지 않은 것은?
주제: 보안
    
질문: 변동성이 큰 시장에서 안정적인 수익을 추구하는 투자 포트폴리오 구성 방법은?
주제: 금융
    
질문: {question}
주제:"""
    
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

# main
if __name__ == "__main__":
    # 1. 벡터DB 초기화
    vdb = LocalVectorDB(model_name=EMBEDDING_MODEL_NAME)

    # 2. FAISS INDEX 파일 로드 또는 생성
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOC_META_PATH):
        print(f"FAISS 인덱스 파일({FAISS_INDEX_PATH})이 존재하지 않아 새로 생성합니다.")
        doc_processor = DocumentProcessor()
        pdf_files = glob.glob(os.path.join(PDF_PATH, "*.pdf"))
        all_chunks = []
        for pdf_file in tqdm(pdf_files, desc="PDF 파일 처리 중"):
            try:
                chunks = doc_processor.preprocess(file_path=pdf_file)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
        if all_chunks:
            vdb.create_and_save_index(all_chunks, FAISS_INDEX_PATH, DOC_META_PATH)
        else:
            print("처리할 문서가 없습니다. PDF 파일 경로를 확인하세요.")
            exit()
    else:
        print("기존 FAISS 인덱스를 로드합니다.")
        vdb.load_index(FAISS_INDEX_PATH, DOC_META_PATH)

    # 3. LLM 모델 다운로드 및 로드
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
        
    print("기본 모델과 LoRA 어댑터 모델을 모두 로드합니다...")
    # 1. 토크나이저 로드 (LoRA 튜닝된 토크나이저를 공통으로 사용)
    tokenizer = AutoTokenizer.from_pretrained(LORA_ASSETS_PATH)

    # 2. 기본 모델 및 파이프라인 생성 (법률 RAG용)
    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )
    base_pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    print("-> 기본 모델 파이프라인 로드 완료.")

    # 3. LoRA 어댑터 적용 모델 및 파이프라인 생성 (금융/보안용)
    lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    lora_pipe = pipeline(
        "text-generation",
        model=lora_model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    print("-> LoRA 적용 모델 파이프라인 로드 완료.")

    # 4. 추론 시작 - 조건부 RAG
    print("조건부 RAG 추론을 시작합니다...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    preds = []

    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="RAG 추론 중"):
        question = row['Question']
        # 질문 분류는 기본 모델을 사용
        category = classify_question_with_llm(question, base_pipe)
        print(f"\n질문 ID: {row['ID']}, 분류: {category}")

        if category == "법률":
            retrieved_docs = vdb.search(question, k=3)
            prompt = make_rag_prompt(question, retrieved_docs)
            # RAG 파라미터 (기존 설정 복원)
            if is_multiple_choice(question):
                generation_kwargs = {"max_new_tokens": 5, "do_sample": False}
            else:
                generation_kwargs = {"max_new_tokens": 150, "temperature": 0.1, "top_p": 0.95, "do_sample": True}
            # 법률 문제는 기본 모델 사용
            output = base_pipe(prompt, **generation_kwargs)

        else: # 금융, 보안
            prompt = make_baseline_prompt(question) # baseline 프롬프트 사용
            # baseline 파라미터 (사용자 요청으로 수정)
            generation_kwargs = {"max_new_tokens": 128, "temperature": 0.2, "top_p": 0.9, "do_sample": True}
            # 금융/보안 문제는 LoRA 모델 사용
            output = lora_pipe(prompt, **generation_kwargs)
        
        pred_answer = extract_answer_only(output[0]["generated_text"], original_question=question)
        preds.append(pred_answer)

    print("추론 완료~!")

    # 5. 제출 파일 생성
    print("제출 파일을 생성합니다.")
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission_df['Answer'] = preds
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"성공적으로 '{SUBMISSION_CSV_PATH}' 파일을 생성하였습니다.")