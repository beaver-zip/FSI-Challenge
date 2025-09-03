# -*- coding: utf-8 -*-
"""
TAPT on FinShibainu MCQA (only) continuing from a DAPT LoRA adapter.
- MCQA 전처리: options A~Z / 1) / 2. → 1.~N. 표준화, answer(문자/숫자) → '1'~'N' 정규화(오직 MCQA에만)
- TAPT: 기본은 정답 미포함(CLМ). --include_answers 로 정답 포함 가능(SFT 유사)
- 이어학습: --init_adapter_from 으로 DAPT LoRA를 초기값으로 불러와 계속 학습
"""

import os, re, gc, math, argparse, inspect, types
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer,
    set_seed, BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- 기본 유틸 ----------------
def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

def ensure_cache_dirs(cache_dir: str, tmp_dir: str | None = None):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for sub in ("hub", "datasets", "transformers"):
            os.makedirs(os.path.join(cache_dir, sub), exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
        os.environ.pop("TRANSFORMERS_CACHE", None)
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        os.environ["TMPDIR"] = tmp_dir

def smart_tokenizer(model_name: str, cache_dir=None):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

# ---------------- MCQA 전처리 ----------------
_re_num_pref = re.compile(r"^\s*([0-9]+)[\.\)]\s*")   # 1. / 2) ...
_re_let_pref = re.compile(r"^\s*([A-Za-z])[\.\)]\s*") # A. / b) ...

def _strip_prefix(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _re_num_pref.sub("", s)
    s = _re_let_pref.sub("", s)
    return s.strip()

def format_options_to_1N(opts):
    """
    options → ['1. foo','2. bar', ...]
    - 리스트/튜플/줄바꿈 문자열 모두 처리
    - 각 항목의 선행 접두어(문자/숫자)를 제거 후 1~N 재부여
    """
    if opts is None:
        return []
    if isinstance(opts, (list, tuple)):
        items = [str(x) for x in opts if str(x).strip()]
    else:
        items = [x for x in str(opts).splitlines() if x.strip()]
    normalized = []
    for i, raw in enumerate(items):
        clean = _strip_prefix(raw)
        normalized.append(f"{i+1}. {clean}")
    return normalized

def map_mcqa_answer_to_1N(ans, num_options: int):
    """
    MCQA 정답을 '1'~'N'으로 표준화.
    - 'A'~'Z'/'a'~'z' → 1~26 (N 초과면 원본 유지)
    - '1'~'N' 문자열/정수는 유효성만 확인
    - 그 외(텍스트 등)는 그대로 반환
    """
    if ans is None:
        return None

    def _letter_to_index(ch: str) -> int | None:
        if len(ch) != 1: return None
        if 'A' <= ch <= 'Z': return ord(ch) - ord('A') + 1
        if 'a' <= ch <= 'z': return ord(ch) - ord('a') + 1
        return None

    if isinstance(ans, str):
        s = ans.strip()
        if len(s) == 1:
            li = _letter_to_index(s)
            if li is not None:
                return str(li) if 1 <= li <= num_options else s
            if s.isdigit():
                v = int(s)
                return str(v) if 1 <= v <= num_options else s
        # 'Answer: C' 같은 꼴 방어
        last = s.split()[-1]
        li = _letter_to_index(last)
        if li is not None:
            return str(li) if 1 <= li <= num_options else s
        if last.isdigit():
            v = int(last)
            return str(v) if 1 <= v <= num_options else s
        return s

    if isinstance(ans, int):
        return str(ans) if 1 <= ans <= num_options else str(ans)

    return str(ans)

def preprocess_mcqa_example(ex):
    """
    MCQA 샘플 전처리:
      - options → 1./2./.../N. 리스트
      - answer  → '1'~'N' (MCQA 전용)
    """
    opts = ex.get("options")
    norm_opts = format_options_to_1N(opts)
    ex["options"] = norm_opts
    ex["answer"] = map_mcqa_answer_to_1N(ex.get("answer"), num_options=len(norm_opts) or 4)
    # type 힌트(선택)
    ex["type"] = ex.get("type") or "mcqa"
    return ex

# ---------------- 직렬화(텍스트 생성) ----------------
def mcqa_to_text(ex, include_answers=False):
    q = ex.get("question") or ""
    opts = ex.get("options") or []
    opts_str = "\n".join(opts) if isinstance(opts, (list, tuple)) else "\n".join(format_options_to_1N(opts))
    s = f"### MCQA\nQuestion: {q}\nOptions:\n{opts_str}\n"
    if include_answers and ex.get("answer") is not None:
        s += f"Answer: {ex['answer']}\n"
    return s

# ---------------- 토크나이즈/블록 ----------------
def make_chunkers(tokenizer, block_size: int):
    def tok_fn(examples):
        return tokenizer(examples["text"])
    def group_fn(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concatenated["input_ids"])
        total = (total // block_size) * block_size
        out = {
            k: [concatenated[k][i:i+block_size] for i in range(0, total, block_size)]
            for k in concatenated.keys()
        }
        out["labels"] = out["input_ids"].copy()
        return out
    return tok_fn, group_fn

# ---------------- Trainer 빌드(버전 호환) ----------------
def build_trainer(model, tokenizer, train_ds, eval_ds, out_dir,
                  lr=1e-5, epochs=1.0, bs=1, grad_acc=32,
                  warmup=0.03, weight_decay=0.1, save_steps=500, log_steps=50):
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    has_eval_strategy_old = "evaluation_strategy" in params   # ≤4.54
    has_eval_strategy_new = "eval_strategy" in params         # 4.55+

    ta = dict(
        output_dir=out_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_acc,
        warmup_ratio=warmup,
        weight_decay=weight_decay,
        logging_steps=log_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=(get_dtype()==torch.bfloat16),
        fp16=(get_dtype()==torch.float16),
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        **({"place_model_on_device": False} if "place_model_on_device" in params else {})
    )
    if eval_ds is not None:
        if has_eval_strategy_new:   ta["eval_strategy"] = "steps"
        elif has_eval_strategy_old: ta["evaluation_strategy"] = "steps"
        if "eval_steps" in params:  ta["eval_steps"] = max(save_steps, log_steps)

    args = TrainingArguments(**ta)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=collator)

    # Transformers 4.55.x: device_map 모델 이동 no-op 패치
    if "place_model_on_device" not in params and getattr(model, "hf_device_map", None) is not None:
        def _noop_move(self, model, device): return model
        trainer._move_model_to_device = types.MethodType(_noop_move, trainer)
    return trainer

# ---------------- 모델 로드 ----------------
def load_base_resume(model_name: str, dtype, device_map="auto",
                     load_in_4bit=False, cache_dir: str | None = None):
    """
    ① 로컬 캐시만으로 먼저 시도
    ② 실패 시 온라인 폴백(동일 cache_dir로 이어받기)
    """
    kwargs = dict(torch_dtype=dtype, device_map=device_map, cache_dir=cache_dir)
    if load_in_4bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            )
        except Exception:
            print("[WARN] bitsandbytes 미설치 → 4bit 비활성.")

    # 1) 로컬 전용
    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True, **kwargs
        )
        print("[load] Base model loaded from local cache only")
        return base
    except Exception as e:
        print(f"[load] Local-only base load failed ({e.__class__.__name__}). "
              f"Falling back to online resume …")
        # 2) 온라인 이어받기(같은 cache_dir 사용 → 중복 다운로드 방지/이어받기)
        base = AutoModelForCausalLM.from_pretrained(
            model_name, **kwargs
        )
        print("[load] Base model downloaded/resumed to cache")
        return base

def continue_from_adapter(base_model, adapter_path: str, is_kbit=False):
    if is_kbit:
        base_model = prepare_model_for_kbit_training(base_model)
    # DAPT 어댑터를 trainable 상태로 로드 → TAPT 계속 학습
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.print_trainable_parameters()
    return model

def load_tokenizer_resume(model_name: str, cache_dir: str):
    """
    ① 로컬 캐시만(local_files_only=True)으로 먼저 시도
    ② 실패 시 온라인 폴백(동일 cache_dir로 이어받기)
    """
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, cache_dir=cache_dir, local_files_only=True
        )
        print("[load] Tokenizer loaded from local cache only")
        return tok
    except Exception as e:
        print(f"[load] Local-only tokenizer load failed ({e.__class__.__name__}). "
              f"Falling back to online resume …")
        tok = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, cache_dir=cache_dir  # 온라인 이어받기
        )
        print("[load] Tokenizer downloaded/resumed to cache")
        return tok

# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()
    # Base / 어댑터
    ap.add_argument("--model_name", type=str, default="K-intelligence/Midm-2.0-Base-Instruct")
    ap.add_argument("--init_adapter_from", type=str, required=True,
                    help="DAPT LoRA 어댑터 경로. 예: /workspace/main/ckpt_dapt_lora/lora_adapter")
    ap.add_argument("--output_dir", type=str, default="ckpt_tapt_lora")
    ap.add_argument("--seed", type=int, default=42)

    # Cache/Temp
    ap.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    ap.add_argument("--tmp_dir", type=str, default=os.environ.get("TMPDIR", "/workspace/tmp"))

    # Data
    ap.add_argument("--include_answers", action="store_true",
                    help="TAPT에도 정답을 포함(SFT 유사). 기본은 미포함(순수 CLM)")
    ap.add_argument("--val_ratio", type=float, default=0.0)
    ap.add_argument("--block_size", type=int, default=2048)

    # Train hparams
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)

    # QLoRA
    ap.add_argument("--load_in_4bit", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)
    ensure_cache_dirs(args.cache_dir, args.tmp_dir)

    assert os.path.isdir(args.init_adapter_from), f"adapter not found: {args.init_adapter_from}"

    # 1) 데이터 로드(오직 mcqa)
    ds = load_dataset("aiqwe/FinShibainu", name="mcqa", cache_dir=args.cache_dir)
    train_raw = ds["train"].map(preprocess_mcqa_example)

    # 2) 직렬화: TAPT -> 정답 미포함(기본)
    def to_text(ex):
        return {"text": mcqa_to_text(ex, include_answers=args.include_answers)}

    train_txt = train_raw.map(to_text, remove_columns=[c for c in train_raw.column_names if c != "type"])

    # 검증셋 옵션
    if args.val_ratio and args.val_ratio > 0.0:
        split = train_txt.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_txt, eval_txt = split["train"], split["test"]
    else:
        eval_txt = None

    # 3) 토크나이즈 & 고정 길이 블록화
    tok = load_tokenizer_resume(args.model_name, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    eff_block = min(args.block_size, getattr(tok, "model_max_length", 10**9))
    tok_fn, grp_fn = make_chunkers(tok, eff_block)
    train_tok = train_txt.map(tok_fn, batched=True, remove_columns=["text","type"])
    train_grp = train_tok.map(grp_fn, batched=True)

    eval_grp = None
    if eval_txt is not None:
        eval_tok = eval_txt.map(tok_fn, batched=True, remove_columns=["text","type"])
        eval_grp = eval_tok.map(grp_fn, batched=True)

    # 4) 베이스 + DAPT LoRA 이어 학습
    dtype = get_dtype()
    device_map = "auto" if torch.cuda.is_available() else None
    base = load_base_resume(args.model_name, dtype=dtype, device_map=device_map,
                        load_in_4bit=args.load_in_4bit, cache_dir=args.cache_dir)
    try:
        base.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    model = continue_from_adapter(base, args.init_adapter_from, is_kbit=args.load_in_4bit)

    # 5) Trainer
    trainer = build_trainer(
        model, tok, train_grp, eval_grp, args.output_dir,
        lr=args.learning_rate, epochs=args.num_train_epochs,
        bs=args.per_device_train_batch_size, grad_acc=args.gradient_accumulation_steps,
        warmup=args.warmup_ratio, weight_decay=args.weight_decay,
        save_steps=args.save_steps, log_steps=args.logging_steps
    )

    # 6) 학습
    trainer.train()

    # 7) 저장(LoRA 어댑터만)
    os.makedirs(args.output_dir, exist_ok=True)
    out_adapter = os.path.join(args.output_dir, "lora_adapter")
    trainer.model.save_pretrained(out_adapter)
    tok.save_pretrained(args.output_dir)
    print(f"[INFO] LoRA adapter saved → {out_adapter}")

    # 8) 선택: 간단 평가(ppl)
    if eval_grp is not None:
        try:
            del trainer.optimizer, trainer.lr_scheduler
        except Exception:
            pass
        gc.collect(); torch.cuda.empty_cache()
        if hasattr(trainer.model, "config"):
            trainer.model.config.use_cache = False
        metrics = trainer.evaluate()
        try:
            ppl = math.exp(metrics["eval_loss"])
        except Exception:
            ppl = float("nan")
        print(f"[INFO] eval_loss={metrics.get('eval_loss'):.4f}, ppl={ppl:.2f}")

if __name__ == "__main__":
    main()
