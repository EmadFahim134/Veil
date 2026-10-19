#!/usr/bin/env python3
"""
Veil - export a trained model to GGUF (no retraining).

Loads your published LoRA adapters (e.g. YourUser/Veil-0.5B-lora), merges them
into the base model and converts the result to GGUF for llama.cpp, Ollama and
LM Studio.

Run it on a Colab GPU:
    pip install unsloth huggingface_hub
    python export_gguf.py --adapter YourUser/Veil-0.5B-lora
    python export_gguf.py --adapter YourUser/Veil-0.5B-lora --push-to-hub --hf-repo YourUser/Veil-0.5B-GGUF

Quantization choices (size for a ~0.5B model, roughly):
    q4_k_m  ~0.4 GB  best size/quality balance (default)
    q8_0    ~0.5 GB  higher quality
    f16     ~1.0 GB  no quantization
"""

from unsloth import FastLanguageModel

import argparse
import glob
import os


def load_env_file(path=".env"):
    """Tiny .env reader (HF_TOKEN or API_KEY)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.replace("export ", "").strip(),
                                  value.strip().strip('"').strip("'"))
    if "HF_TOKEN" not in os.environ and "API_KEY" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["API_KEY"]


def main():
    p = argparse.ArgumentParser(description="Export Veil to GGUF.")
    p.add_argument("--adapter", required=True,
                   help="HF repo or local folder with the LoRA adapters, e.g. YourUser/Veil-0.5B-lora")
    p.add_argument("--quant", default="q4_k_m",
                   help="Quantization method, or several separated by commas (e.g. q4_k_m,q8_0)")
    p.add_argument("--out-dir", default="outputs/veil-gguf")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hf-repo", default=None, help="e.g. YourUser/Veil-0.5B-GGUF")
    args = p.parse_args()

    load_env_file()
    token = os.environ.get("HF_TOKEN")
    if args.push_to_hub and not (args.hf_repo and token):
        raise SystemExit("--push-to-hub needs --hf-repo and the HF_TOKEN environment variable.")

    quants = [q.strip() for q in args.quant.split(",") if q.strip()]
    quant_arg = quants[0] if len(quants) == 1 else quants

    # Load the adapters; Unsloth fetches the base model they were trained on.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    # Merge to 16-bit and convert (this builds llama.cpp the first time, a few minutes).
    model.save_pretrained_gguf(args.out_dir, tokenizer, quantization_method=quant_arg)

    # Unsloth may write to <out-dir> or <out-dir>_gguf depending on its version.
    found = []
    for folder in (args.out_dir, args.out_dir + "_gguf"):
        found += glob.glob(os.path.join(folder, "**", "*.gguf"), recursive=True)
    found += glob.glob("*.gguf")  # some versions write next to the script
    # llama.cpp ships tiny vocab-only .gguf files; they are not your model.
    found = sorted({f for f in found if "vocab" not in os.path.basename(f).lower()})

    print("\nGGUF files:")
    for path in found:
        print(f"  {path}  ({os.path.getsize(path) / 1e6:.0f} MB)")
    if not found:
        print("  none found - check the log above for errors.")

    if args.push_to_hub:
        model.push_to_hub_gguf(args.hf_repo, tokenizer, quantization_method=quant_arg, token=token)
        print(f"\nPushed to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
