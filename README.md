# Veil

Veil is a small AI model that helps students and teachers **create tests**. Give it a subject and a number of questions, and it writes the questions, the answer options and an answer key.

It is fine-tuned with QLoRA ([Unsloth](https://github.com/unslothai/unsloth)) from a ~0.5B base model, so the 4-bit version is small enough to run in roughly 500-700 MB.

> **Model:** `https://huggingface.co/YOUR_HF_USERNAME/Veil-0.5B` (link added after publishing)

## What it does

- Multiple-choice tests with an answer key
- Short-answer questions written from a passage you provide
- Math word problems with step-by-step solutions (optional dataset)

Example of the output format Veil is trained to produce:

```text
Create a 2-question multiple-choice test on science with an answer key.

1. Which gas do plants take in for photosynthesis?
A) Oxygen
B) Carbon dioxide
C) Nitrogen
D) Hydrogen

2. Where does photosynthesis mainly take place in a plant cell?
A) Nucleus
B) Mitochondria
C) Chloroplast
D) Ribosome

Answer key:
1. B
2. C
```

## Build your own

Want to train a model like this for your own subject or school? Follow the full guide:

**[docs/BUILD-YOUR-OWN.md](docs/BUILD-YOUR-OWN.md)**

It covers setup with uv, loading your Hugging Face token, downloading data, training on Colab, and publishing.

## What is in this repo

| File | Purpose |
|---|---|
| `train.py` | QLoRA training script (run on a GPU, e.g. Colab) |
| `get-veil-data.ps1` | Downloads and formats a small training set |
| `load.ps1` | Loads your `.env` (Hugging Face token) into PowerShell |
| `.env.example` | Template for your token (copy to `.env`, never commit it) |
| `data/train.jsonl` | Generated training data |

## Limitations

Small models make mistakes. Generated questions and answer keys can be wrong, so a teacher should review every test before using it.

## License

The code is released under **GPL-3.0** (see [LICENSE](LICENSE)). The model weights and training data have their own licenses. See the license section of the [build guide](docs/BUILD-YOUR-OWN.md#licenses) before you reuse or redistribute them.
