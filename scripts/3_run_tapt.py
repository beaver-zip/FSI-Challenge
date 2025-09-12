# -*- coding: utf-8 -*-
"""
TAPT on FinShibainu MCQA, continuing from a DAPT LoRA adapter.
- 모델: models/base_model (로컬)
- 입력 데이터: data/raw/mcqa_dataset (로컬)
- 입력 어댑터: models/dapt_adapter/lora_adapter
- 출력: models/tapt_adapter
"""

import os, re, gc, math, argparse, inspect, types
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer,
    set_seed, BitsAndBytesConfig
)
from peft import PeftModel, prepare_model_for_kbit_training

# --- 이하 모든 함수는 원본과 동일하게 유지 ---

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
    tok.padding_side = "right"
    return tok

_re_num_pref = re.compile(r"^\s*([0-9]+)[\.\)]\s*")
_re_let_pref = re.compile(r"^\s*([A-Za-z])[\.\)]\s*")

def _strip_prefix(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _re_num_pref.sub("", s)
    s = _re_let_pref.sub("", s)
    return s.strip()

def format_options_to_1N(opts):
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
    opts = ex.get("options")
    norm_opts = format_options_to_1N(opts)
    ex["options"] = norm_opts
    ex["answer"] = map_mcqa_answer_to_1N(ex.get("answer"), num_options=len(norm_opts) or 4)
    ex["type"] = ex.get("type") or "mcqa"
    return ex

def mcqa_to_text(ex, include_answers=False):
    q = ex.get("question") or ""
    opts = ex.get("options") or []
    opts_str = "\n".join(opts) if isinstance(opts, (list, tuple)) else "\n".join(format_options_to_1N(opts))
    s = f"### MCQA\nQuestion: {q}\nOptions:\n{opts_str}\n"
    if include_answers and ex.get("answer") is not None:
        s += f"Answer: {ex['answer']}\n"
    return s

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

def build_trainer(model, tokenizer, train_ds, eval_ds, out_dir,
                  lr=1e-5, epochs=1.0, bs=1, grad_acc=32,
                  warmup=0.03, weight_decay=0.1, save_steps=500, log_steps=50):
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    has_eval_strategy_old = "evaluation_strategy" in params
    has_eval_strategy_new = "eval_strategy" in params
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
    if "place_model_on_device" not in params and getattr(model, "hf_device_map", None) is not None:
        def _noop_move(self, model, device): return model
        trainer._move_model_to_device = types.MethodType(_noop_move, trainer)
    return trainer

def load_base_model(model_name: str, dtype, device_map="auto", load_in_4bit=False):
    kwargs = dict(torch_dtype=dtype, device_map=device_map)
    if load_in_4bit:
        try:
            import bitsandbytes as bnb
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
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

def continue_from_adapter(base_model, adapter_path: str, is_kbit=False):
    if is_kbit:
        base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.print_trainable_parameters()
    return model

def main():
    ap = argparse.ArgumentParser()
    # --- 경로 인자 수정 ---
    ap.add_argument("--model_name", type=str, default="models/base_model")
    ap.add_argument("--dataset_path", type=str, default="data/raw/mcqa_dataset")
    ap.add_argument("--init_adapter_from", type=str, default="models/dapt_adapter/lora_adapter")
    ap.add_argument("--output_dir", type=str, default="models/tapt_adapter")
    
    # --- 나머지 인자는 원본 유지 ---
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_answers", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.0)
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--load_in_4bit", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    assert os.path.isdir(args.init_adapter_from), f"adapter not found: {args.init_adapter_from}"

    # --- 데이터 로드 방식 수정 ---
    # Hub가 아닌 로컬 data_dir에서 직접 로드합니다.
    ds = load_dataset("aiqwe/FinShibainu", name="mcqa", data_dir=args.dataset_path)
    train_raw = ds["train"].map(preprocess_mcqa_example)

    def to_text(ex):
        return {"text": mcqa_to_text(ex, include_answers=args.include_answers)}

    train_txt = train_raw.map(to_text, remove_columns=[c for c in train_raw.column_names if c != "type"])

    if args.val_ratio and args.val_ratio > 0.0:
        split = train_txt.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_txt, eval_txt = split["train"], split["test"]
    else:
        eval_txt = None

    tok = smart_tokenizer(args.model_name)
    eff_block = min(args.block_size, getattr(tok, "model_max_length", 10**9))
    tok_fn, grp_fn = make_chunkers(tok, eff_block)
    train_tok = train_txt.map(tok_fn, batched=True, remove_columns=["text","type"])
    train_grp = train_tok.map(grp_fn, batched=True)

    eval_grp = None
    if eval_txt is not None:
        eval_tok = eval_txt.map(tok_fn, batched=True, remove_columns=["text","type"])
        eval_grp = eval_tok.map(grp_fn, batched=True)

    dtype = get_dtype()
    device_map = "auto" if torch.cuda.is_available() else None
    # `from_pretrained`는 로컬 경로를 자동으로 인식합니다.
    base = load_base_model(args.model_name, dtype=dtype, device_map=device_map, load_in_4bit=args.load_in_4bit)
    try:
        base.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    model = continue_from_adapter(base, args.init_adapter_from, is_kbit=args.load_in_4bit)

    trainer = build_trainer(
        model, tok, train_grp, eval_grp, args.output_dir,
        lr=args.learning_rate, epochs=args.num_train_epochs,
        bs=args.per_device_train_batch_size, grad_acc=args.gradient_accumulation_steps,
        warmup=args.warmup_ratio, weight_decay=args.weight_decay,
        save_steps=args.save_steps, log_steps=args.logging_steps
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    out_adapter = os.path.join(args.output_dir, "lora_adapter")
    trainer.model.save_pretrained(out_adapter)
    tok.save_pretrained(args.output_dir)
    print(f"[INFO] LoRA adapter saved → {out_adapter}")

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
