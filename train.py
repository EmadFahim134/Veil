#!/usr/bin/env python3
"""
Veil - QLoRA fine-tuning script (Unsloth + TRL)

Fine-tunes a small instruction model so it can help students/teachers create
tests (MCQs, short answers, quizzes, answer keys).

Install (Colab, GPU runtime such as a free T4):
    pip install unsloth datasets trl huggingface_hub

Training data (JSONL, one example per line). Either format works:

  1) Instruction format
     {"instruction": "Create 3 MCQs on photosynthesis for grade 9",
      "input": "",
      "output": "1. Which gas do plants absorb ...\\nA) ...\\nAnswer: B\\n..."}

  2) Chat format
     {"messages": [{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}]}

Examples:
    python train.py --data data/train.jsonl
    python train.py --data data/train.jsonl --push-to-hub --hf-repo YourUser/Veil-0.5B
    python train.py --data data/train.jsonl --gguf          # also export a GGUF (q4_k_m)

For pushing, set your token first:   export HF_TOKEN=hf_xxx   (or use Colab secrets)
"""

# Unsloth must be imported before transformers / trl.
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

import argparse
import inspect
import os

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

SYSTEM_PROMPT = (
    "You are Veil, an AI assistant that helps students and teachers create "
    "clear, accurate tests: multiple-choice questions, short-answer questions, "
    "true/false, and answer keys. Match the requested topic, grade level and "
    "number of questions. Always include the correct answers."
)

SEED = 3407


def load_env_file(path=".env"):
    """Tiny .env reader so local runs work without exporting variables by hand."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.replace("export ", "").strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    # Accept API_KEY as an alias for HF_TOKEN
    if "HF_TOKEN" not in os.environ and "API_KEY" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["API_KEY"]


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Veil with QLoRA (Unsloth).")
    p.add_argument("--data", required=True, help="Path to training JSONL file")
    p.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct",
                   help="Base model. ~0.5B stays well under 700 MB in 4-bit. "
                        "Try unsloth/Qwen2.5-1.5B-Instruct for better quality (bigger).")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--val-split", type=float, default=0.05,
                   help="Fraction held out for validation (used only if >= 50 examples)")
    p.add_argument("--no-responses-only", action="store_true",
                   help="Train on the full text instead of only the assistant answers")
    p.add_argument("--merge", choices=["none", "4bit", "16bit"], default="4bit",
                   help="Save a merged model (4bit keeps it small)")
    p.add_argument("--gguf", action="store_true", help="Also export GGUF q4_k_m")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hf-repo", default=None, help="e.g. YourUser/Veil-0.5B")
    return p.parse_args()


def to_messages(example):
    """Convert one raw example into a list of chat messages."""
    if example.get("messages"):
        msgs = list(example["messages"])
        if msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
        return msgs

    user = example["instruction"].strip()
    if example.get("input"):
        user += "\n\n" + example["input"].strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"].strip()},
    ]


def response_markers(model_name):
    """Chat-template markers used to mask everything except the answer."""
    if "llama" in model_name.lower():
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    # ChatML (Qwen and others)
    return "<|im_start|>user\n", "<|im_start|>assistant\n"


def main():
    load_env_file()
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")

    if args.push_to_hub and not (args.hf_repo and hf_token):
        raise SystemExit("--push-to-hub needs --hf-repo and the HF_TOKEN environment variable.")

    # 1) Load the base model in 4-bit and attach LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # 2) Load and format the dataset
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"Loaded {len(ds)} examples")

    def fmt(example):
        text = tokenizer.apply_chat_template(
            to_messages(example), tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    train_ds, eval_ds = ds, None
    if len(ds) >= 50 and args.val_split > 0:
        split = ds.train_test_split(test_size=args.val_split, seed=SEED)
        train_ds, eval_ds = split["train"], split["test"]

    # 3) Trainer (kwargs adapt to different TRL versions)
    cfg_kwargs = dict(
        output_dir=args.output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        seed=SEED,
        report_to="none",
        packing=False,
    )
    cfg_params = inspect.signature(SFTConfig.__init__).parameters
    cfg_kwargs["max_length" if "max_length" in cfg_params else "max_seq_length"] = args.max_seq_len

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**cfg_kwargs),
    )
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    if not args.no_responses_only:
        instruction_part, response_part = response_markers(args.model)
        trainer = train_on_responses_only(
            trainer, instruction_part=instruction_part, response_part=response_part
        )

    # 4) Train
    trainer.train()

    # 5) Save LoRA adapters
    adapter_dir = os.path.join(args.output_dir, "veil-lora")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapters to {adapter_dir}")

    # 6) Quick sanity check
    FastLanguageModel.for_inference(model)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Create 3 multiple-choice questions on photosynthesis for grade 9, with answers."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9)
    print("\n--- Sample output ---")
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    # 7) Merged model (what most people download and run)
    method = {"4bit": "merged_4bit_forced", "16bit": "merged_16bit"}.get(args.merge)
    if method:
        merged_dir = os.path.join(args.output_dir, f"veil-merged-{args.merge}")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method=method)
        print(f"Saved merged model to {merged_dir}")
        if args.push_to_hub:
            model.push_to_hub_merged(args.hf_repo, tokenizer, save_method=method, token=hf_token)

    # 8) Optional GGUF for llama.cpp / Ollama / phones
    if args.gguf:
        gguf_dir = os.path.join(args.output_dir, "veil-gguf")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
        if args.push_to_hub:
            model.push_to_hub_gguf(args.hf_repo + "-GGUF", tokenizer,
                                   quantization_method="q4_k_m", token=hf_token)

    # 9) Adapters on the Hub too (small, handy for further training)
    if args.push_to_hub:
        model.push_to_hub(args.hf_repo + "-lora", token=hf_token)
        tokenizer.push_to_hub(args.hf_repo + "-lora", token=hf_token)
        print(f"Pushed to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
