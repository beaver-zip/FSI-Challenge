# -*- coding: utf-8 -*-
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# 임시로 이전 경로를 유지합니다. 다음 단계에서 src 폴더로 옮기고 수정할 예정입니다.
from src.rag import LocalVectorDB

# --- 1. 경로 설정 (상대 경로 기준) ---
# 이 파일의 위치를 기준으로 절대 경로를 계산합니다.
CWD = os.path.dirname(os.path.abspath(__file__))

# 모델 및 DB 경로
BASE_MODEL_PATH = os.path.join(CWD, "models", "base_model")
ADAPTER_PATH = os.path.join(CWD, "models", "tapt_adapter")
FAISS_INDEX_PATH = os.path.join(CWD, "db", "faiss_index.bin")
DOC_META_PATH = os.path.join(CWD, "db", "doc_metadata.pkl")
EMBEDDING_MODEL_NAME = "upskyy/bge-m3-korean" # 이 모델은 VDB 생성 시에만 필요합니다.

# --- 2. 핵심 로직 함수 (hanseo/RAG/run.py에서 그대로 가져옴) ---

def is_multiple_choice(question_text):
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text):
    lines = full_text.strip().split("\n")
    q_lines, options = [], []
    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    return " ".join(q_lines), options

def extract_answer_only(generated_text: str, original_question: str) -> str:
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()
    if not text:
        return "미응답"
    if is_multiple_choice(original_question):
        match = re.match(r"\D*([1-9][0-9]?)", text)
        return match.group(1) if match else "0"
    return text

def make_rag_prompt(question_text, context):
    question, options = extract_question_and_choices(question_text)
    context_str = "\n\n".join(context)
    if is_multiple_choice(question_text):
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
    else:
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

def make_baseline_prompt(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        return (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        return (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 **네 문장 이내로 정확하고 간결하게** 답변하세요.\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )

def classify_question_with_llm(question: str, pipe) -> str:
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
    if "보안" in answer: return "보안"
    if "금융" in answer: return "금융"
    return "법률"

# --- 3. 모델 및 DB 전역 로딩 (최초 1회 실행) ---

print("모델과 벡터DB를 로드합니다. 시간이 다소 걸릴 수 있습니다...")

# 3-1. 벡터DB 로드
vdb = LocalVectorDB(model_name=EMBEDDING_MODEL_NAME)
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOC_META_PATH):
    vdb.load_index(FAISS_INDEX_PATH, DOC_META_PATH)
    print("-> 벡터DB 로드 완료.")
else:
    # 실제 추론 환경에서는 DB가 이미 빌드되어 있어야 함
    vdb = None
    print("[경고] 벡터DB 파일이 존재하지 않습니다. '법률' 질문에 대한 RAG 추론이 불가능합니다.")

# 3-2. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# 3-3. 기본 모델 및 파이프라인 생성 (법률 RAG용)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)
base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer, device_map="auto")
print("-> 기본 모델 파이프라인 로드 완료.")

# 3-4. LoRA 어댑터 적용 모델 및 파이프라인 생성 (금융/보안용)
lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
lora_pipe = pipeline("text-generation", model=lora_model, tokenizer=tokenizer, device_map="auto")
print("-> LoRA 적용 모델 파이프라인 로드 완료.")

print("모든 모델과 리소스 로딩이 완료되었습니다.")


# --- 4. 추론 함수 (streamlit_app.py에서 호출) ---

def infer(question: str) -> str:
    """
    사용자의 질문 한 개를 입력받아 분류 후, 적절한 모델로 추론하여 답변을 반환합니다.
    """
    # 질문 분류 (기본 모델 사용)
    category = classify_question_with_llm(question, base_pipe)
    print(f"질문 분류 결과: {category}")

    if category == "법률":
        if vdb is None:
            return "오류: 벡터DB가 로드되지 않아 법률 질문에 답변할 수 없습니다."
        
        retrieved_docs = vdb.search(question, k=3)
        prompt = make_rag_prompt(question, retrieved_docs)
        
        if is_multiple_choice(question):
            generation_kwargs = {"max_new_tokens": 5, "do_sample": False}
        else:
            generation_kwargs = {"max_new_tokens": 150, "temperature": 0.1, "top_p": 0.95, "do_sample": True}
        
        # 법률 문제는 기본 모델 사용
        output = base_pipe(prompt, **generation_kwargs)

    else: # 금융, 보안
        prompt = make_baseline_prompt(question)
        generation_kwargs = {"max_new_tokens": 128, "temperature": 0.2, "top_p": 0.9, "do_sample": True}
        
        # 금융/보안 문제는 LoRA 모델 사용
        output = lora_pipe(prompt, **generation_kwargs)
    
    # 최종 답변 추출
    pred_answer = extract_answer_only(output[0]["generated_text"], original_question=question)
    return pred_answer
