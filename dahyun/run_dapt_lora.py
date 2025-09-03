# -*- coding: utf-8 -*-
"""
LoRA 기반 DAPT (continued pretraining) on a TXT corpus.
- 입력: output/dapt_corpus.txt (전처리 스크립트가 만든 태그 포함 텍스트)
- 기본: LoRA 사용, 어댑터 저장 (병합 저장 코드는 주석)
- 모델: K-intelligence/Midm-2.0-Base-Instruct (기본값, 변경 가능)
"""

import os
import math
import argparse
import inspect
import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig,
)

# ===== PEFT / LoRA =====
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

def smart_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # CLM 학습은 보통 right padding
    try:
        tok.padding_side = "right"
    except Exception:
        pass
    return tok

def make_chunkers(tokenizer, block_size: int):
    def tok_fn(examples):
        TAG = re.compile(r"<[^>]+>")
        clean = [TAG.sub("", t) for t in examples["text"]]
        return tokenizer(clean)  # no truncation; 이후 그룹핑에서 고정 길이 블록화
    def group_fn(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concatenated["input_ids"])
        total = (total // block_size) * block_size
        result = {
            k: [concatenated[k][i:i+block_size] for i in range(0, total, block_size)]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    return tok_fn, group_fn

def build_trainer(
    model, tokenizer, train_dataset, eval_dataset, out_dir,
    lr=5e-5, epochs=1.0, bs=1, grad_acc=16,
    warmup=0.03, weight_decay=0.0, save_steps=500, log_steps=50
):
    sig = inspect.signature(TrainingArguments.__init__)
    has_eval_strategy_old = "evaluation_strategy" in sig.parameters   # ≤ v4.54
    has_eval_strategy_new = "eval_strategy" in sig.parameters         # v4.55+

    ta_kwargs = dict(
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
    )

    if eval_dataset is not None:
        if has_eval_strategy_new:
            ta_kwargs["eval_strategy"] = "steps"
        elif has_eval_strategy_old:
            ta_kwargs["evaluation_strategy"] = "steps"
        if "eval_steps" in sig.parameters:
            ta_kwargs["eval_steps"] = max(save_steps, log_steps)

    args = TrainingArguments(**ta_kwargs)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return Trainer(model=model, args=args,
                   train_dataset=train_dataset, eval_dataset=eval_dataset,
                   data_collator=collator)

def load_base_model(model_name: str, dtype, device_map="auto", load_in_4bit=False):
    kwargs = dict(torch_dtype=dtype, device_map=device_map)
    if load_in_4bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
            kwargs.update(dict(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ))
        except Exception:
            print("[WARN] bitsandbytes가 없어 4bit 로딩을 건너뜁니다.")
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

# 모델 계열별 권장 타깃
LORA_PRESETS = {
    "attn_only": ("q_proj","k_proj","v_proj","o_proj"),  # Llama/Mistral/Qwen2 계열
    "attn_mlp":  ("q_proj","k_proj","v_proj","o_proj","down_proj","up_proj","gate_proj"),
    "opt_attn":  ("q_proj","k_proj","v_proj","out_proj"), # OPT 계열
}

def attach_lora(model,
                r=8,
                alpha=16,
                dropout=0.1,
                is_kbit=False,
                target_modules=("q_proj","k_proj","v_proj","o_proj")):
    if is_kbit:
        model = prepare_model_for_kbit_training(model)
    # gradient checkpointing을 쓸 때는 cache 비활성화 권장
    if hasattr(model, "config"):
        model.config.use_cache = False

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

def save_lora_adapter(trainer, out_dir: str):
    adapter_dir = os.path.join(out_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_dir)
    return adapter_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="K-intelligence/Midm-2.0-Base-Instruct")
    ap.add_argument("--data_file", type=str, default="output/dapt_corpus.txt")
    ap.add_argument("--output_dir", type=str, default="ckpt_dapt_lora")
    ap.add_argument("--seed", type=int, default=42)

    # 데이터/토크나이즈
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--val_ratio", type=float, default=0.02)

    # 학습 하이퍼파라미터
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--num_train_epochs", type=float, default=0.5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)

    # LoRA / QLoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="QLoRA: 4bit 양자화 로드(bitsandbytes 필요)")
    ap.add_argument("--lora_target", type=str, default=None,
                    help="쉼표로 구분된 모듈 목록. 예: q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--lora_preset", type=str, default="attn_only",
                    choices=["attn_only","attn_mlp","opt_attn"],
                    help="모델 계열에 맞는 LoRA 타깃 프리셋")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="이 경로의 체크포인트에서 학습 재개")

    # 저장 관련 (현재는 어댑터만 저장)
    ap.add_argument("--save_adapter", action="store_true",
                    help="훈련 후 LoRA 어댑터 저장 (현재 코드에서는 항상 저장)")

    args = ap.parse_args()
    set_seed(args.seed)

    assert os.path.exists(args.data_file), f"data_file not found: {args.data_file}"

    # ===== Tokenizer & Dataset =====
    tokenizer = smart_tokenizer(args.model_name)

    # block_size를 토크나이저 한계에 맞춰 안전하게 클램프
    effective_block_size = args.block_size
    try:
        max_len = getattr(tokenizer, "model_max_length", None)
        if max_len and isinstance(max_len, int) and max_len < 10**8:
            effective_block_size = min(args.block_size, max_len)
    except Exception:
        pass

    ds = load_dataset("text", data_files=args.data_file)["train"]
    if args.val_ratio and args.val_ratio > 0:
        split = ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_raw, eval_raw = split["train"], split["test"]
    else:
        train_raw, eval_raw = ds, None

    tok_fn, group_fn = make_chunkers(tokenizer, effective_block_size)
    train_tok = train_raw.map(tok_fn, batched=True, remove_columns=["text"])
    train_group = train_tok.map(group_fn, batched=True)

    eval_group = None
    if eval_raw is not None:
        eval_tok = eval_raw.map(tok_fn, batched=True, remove_columns=["text"])
        eval_group = eval_tok.map(group_fn, batched=True)

    # ===== Model (LoRA / QLoRA) =====
    dtype = get_dtype()
    device_map = "auto" if torch.cuda.is_available() else None
    model = load_base_model(args.model_name, dtype=dtype, device_map=device_map, load_in_4bit=args.load_in_4bit)

    # pad_token_id를 모델에도 반영
    try:
        model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    # gradient checkpointing 대비
    if hasattr(model, "config"):
        model.config.use_cache = False

    # LoRA 타깃 결정
    if args.lora_target:
        target_modules = tuple([s.strip() for s in args.lora_target.split(",") if s.strip()])
    else:
        target_modules = LORA_PRESETS[args.lora_preset]

    model = attach_lora(model,
                        r=args.lora_r,
                        alpha=args.lora_alpha,
                        dropout=args.lora_dropout,
                        is_kbit=args.load_in_4bit,
                        target_modules=target_modules)

    # ===== Trainer =====
    trainer = build_trainer(
        model, tokenizer,
        train_group, eval_group, args.output_dir,
        lr=args.learning_rate, epochs=args.num_train_epochs,
        bs=args.per_device_train_batch_size, grad_acc=args.gradient_accumulation_steps,
        warmup=args.warmup_ratio, weight_decay=args.weight_decay,
        save_steps=args.save_steps, log_steps=args.logging_steps
    )

    # ===== Train =====
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ===== Save: 어댑터 저장 =====
    os.makedirs(args.output_dir, exist_ok=True)
    adapter_dir = save_lora_adapter(trainer, args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] LoRA adapter saved to: {adapter_dir}")

    # ===== Quick eval: PPL =====
    if eval_group is not None:
        metrics = trainer.evaluate()
        try:
            ppl = math.exp(metrics["eval_loss"])
        except Exception:
            ppl = float("nan")
        print(f"[INFO] Validation loss: {metrics.get('eval_loss'):.4f}, ppl: {ppl:.2f}")

if __name__ == "__main__":
    main()
