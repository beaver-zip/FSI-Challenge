import os
import re
import glob
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from Preprocessor.preprocessor import DocumentProcessor
from VDB.localVDB import LocalVectorDB

# --- 경로 설정 ---
TEST_CSV_PATH = "test.csv"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
SUBMISSION_CSV_PATH = "submission.csv"

PDF_PATH = "pdfs"
LLM_MODEL_PATH = "/workspace/model"
EMBEDDING_MODEL_NAME = "upskyy/bge-m3-korean"
FAISS_INDEX_PATH = "faiss_index.bin"
DOC_META_PATH = "doc_metadata.pkl"

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
            "당신은 금융보안 전문가입니다.\n"
            "주어진 [참고 자료]를 참고하여, [질문]에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
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
            "당신은 금융보안 전문가입니다.\n"
            "주어진 [참고 자료]를 참고하여, [질문]에 대해 **세 문장 이내로 정확하고 간결하게** 서술하세요.\n"
            "만약 [참고 자료]의 내용이 [질문]과 관련 없거나 부족한 경우, 당신의 전문 지식을 활용하여 답변하세요.\n\n"
            "[참고 자료]\n"
            f"{context_str}\n\n"
            "[질문]\n"
            f"{question_text}\n\n"
            "답변:"
        )

def make_zeroshot_prompt(question_text):
    question, options = extract_question_and_choices(question_text)
    if is_multiple_choice(question_text):
        return (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
            "[질문]\n"
            f"{question}\n\n"
            "[선택지]\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        return (
            "당신은 금융보안 전문가입니다.\n"
            "다음 주관식 질문에 대해 **세 문장 이내로 정확하고 간결하게** 서술하세요.\n\n"
            "[질문]\n"
            f"{question_text}\n\n"
            "답변:"
        )


# LLM 호출 1: 주제 분류 (법률은 pdfs에 있는 문서로 RAG, 보안/금융은 Zero-shot으로 처리)
def classify_question_with_llm(question: str, pipe) -> str:
    classification_prompt = f
    """다음 질문의 주제를 "법률", "금융", "보안" 중 하나로만 답하세요.
    
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

    # 3. LLM 로드
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

    # 4. 추론 시작 - 조건부 RAG
    print("조건부 RAG 추론을 시작합니다...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    preds = []

    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="RAG 추론 중"):
        question = row['Question']
        category = classify_question_with_llm(question, pipe)
        print(f"\n질문 ID: {row['ID']}, 분류: {category}")

        if category == "법률":
            retrieved_docs = vdb.search(question, k=3)
            prompt = make_rag_prompt(question, retrieved_docs)
        else:
            prompt = make_zeroshot_prompt(question)

        # 객관식 / 주관식 답변 파라미터 설정
        if is_multiple_choice(question): # 객관식
            generation_kwargs = {"max_new_tokens": 5, "do_sample": False}
        else: # 주관식
            generation_kwargs = {"max_new_tokens": 150, "temperature": 0.1, "top_p": 0.95, "do_sample": True}

        output = pipe(prompt, **generation_kwargs)
        
        pred_answer = extract_answer_only(output[0]["generated_text"], original_question=question)
        preds.append(pred_answer)

    print("추론 완료~!")

    # 5. 제출 파일 생성
    print("제출 파일을 생성합니다.")
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission_df['Answer'] = preds
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"성공적으로 '{SUBMISSION_CSV_PATH}' 파일을 생성하였습니다.")
