"""
QLoRA Fine-tuning - Pakistan Constitution Q&A
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~2.2GB, fits on 6GB VRAM)
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR        = r"D:\pakistan_constitution_llm"
TRAIN_FILE      = f"{BASE_DIR}\\data\\splits\\constitution_train.jsonl"
VAL_FILE        = f"{BASE_DIR}\\data\\splits\\constitution_validation.jsonl"
CHECKPOINT_DIR  = f"{BASE_DIR}\\training\\checkpoints"
ADAPTER_DIR     = f"{BASE_DIR}\\training\\final_adapter"
MODEL_CACHE_DIR = r"D:\hf_cache"

for d in [CHECKPOINT_DIR, ADAPTER_DIR, MODEL_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ["HF_HOME"]            = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ← FIXED: Real model that exists
MAX_LENGTH   = 512
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LR           = 2e-4
EPOCHS       = 5
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# ── Load JSONL ─────────────────────────────────────────────────────
def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def format_prompt(example: dict) -> str:
    # TinyLlama Chat uses special tokens, but Alpaca format still works
    return (
        "### Instruction:\n"
        "You are a legal assistant specialising in the Pakistan Constitution. "
        "Answer the following question accurately.\n\n"
        f"### Question:\n{example['question']}\n\n"
        f"### Answer:\n{example['answer']}"
    )

# ── Main ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Pakistan Constitution Q&A - QLoRA Training")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # ── 1. Tokenizer ────────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── 2. Model (4-bit) ────────────────────────────────────────────
    print("[2/5] Loading model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
    )

    # ── 3. Prepare for training ───────────────────────────────────
    print("[3/5] Preparing model for training...")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ── 4. LoRA ───────────────────────────────────────────────────
    print("[4/5] Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Dataset ────────────────────────────────────────────────
    print("[5/5] Preparing datasets...")
    train_raw = load_jsonl(TRAIN_FILE)
    val_raw = load_jsonl(VAL_FILE)

    def build_dataset(records):
        texts = [format_prompt(r) for r in records]
        ds = Dataset.from_dict({"text": texts})
        return ds.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = build_dataset(train_raw)
    val_ds = build_dataset(val_raw) if val_raw else None

    print(f"      Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0}")

    # ── 6. Training ─────────────────────────────────────────────────
    print("\n>>> Starting training...")
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=50,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False,
        report_to="none",
        optim="adamw_torch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint()

    # ── 7. Save ─────────────────────────────────────────────────────
    print(f"\n>>> Saving adapter to {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    info = {
        "base_model": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "epochs": EPOCHS,
        "train_examples": len(train_ds),
    }
    with open(f"{ADAPTER_DIR}\\training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n✓ Training complete!")
    print(f"  Adapter saved: {ADAPTER_DIR}")

if __name__ == "__main__":
    main()