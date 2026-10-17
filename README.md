# Veil

Veil is a small AI model that helps students and teachers **create tests**: multiple-choice questions, short-answer questions and answer keys. It is fine-tuned with QLoRA ([Unsloth](https://github.com/unslothai/unsloth)) and is small enough to run in roughly 500-700 MB in 4-bit.

This repo is also a **guide and toolkit for building your own Veil-like model**: download a small dataset, train on a free Colab GPU, and publish to Hugging Face.

> Model: `https://huggingface.co/YOUR_HF_USERNAME/Veil-0.5B` (replace with your link after publishing)

## What you need

- A computer with [uv](https://docs.astral.sh/uv/), [git](https://git-scm.com/) and PowerShell (`pwsh`)
- A free [GitHub](https://github.com) account and a free [Hugging Face](https://huggingface.co) account
- A free [Google Colab](https://colab.research.google.com) account (the training GPU)

Training needs an NVIDIA GPU. If you don't have one, Colab's free T4 is enough for a 0.5B model.

## Project layout

```
Veil/
├── train.py             # QLoRA training script (run on a GPU, e.g. Colab)
├── get-veil-data.ps1    # downloads and formats a small training set
├── load.ps1             # loads your .env (Hugging Face token) into PowerShell
├── .env.example         # template for your token (copy to .env)
├── data/train.jsonl     # generated training data
└── pyproject.toml       # uv project
```

## Build your own, step by step

### 1. Set up the project

```powershell
git clone https://github.com/EmadFahim134/Veil.git
cd Veil
uv sync
```

### 2. Add your Hugging Face token

Create a token with **Write** access at <https://huggingface.co/settings/tokens>, then:

```powershell
Copy-Item .env.example .env
# open .env and paste your token after HF_TOKEN=
Add-Content .gitignore "`n.env"
./load.ps1
```

`.env` holds a secret. It must be in `.gitignore` so it is never committed. `load.ps1` loads it into your terminal session and warns you if it is not ignored.

### 3. Download the training data

```powershell
./get-veil-data.ps1
```

This pulls only the rows it needs (about 8,000 by default, a few MB) from public question datasets and writes `data/train.jsonl`. Useful options:

```powershell
./get-veil-data.ps1 -Sciq 1000 -Arc 500 -OpenBook 500 -Squad 800 -Math 600
./get-veil-data.ps1 -Sciq 0     # skip a source (see the license note below)
```

Check the result:

```powershell
(Get-Content data/train.jsonl | Measure-Object -Line).Lines
Get-Content data/train.jsonl -TotalCount 1 | ConvertFrom-Json | Format-List
```

Each line looks like this:

```json
{"instruction": "Create a 4-question multiple-choice test on science with an answer key.", "input": "", "output": "1. ...\nA) ...\n\nAnswer key:\n1. B"}
```

### 4. Push to GitHub

```powershell
git add .
git commit -m "Add data and training scripts"
git push origin main
```

### 5. Train on Colab

Open a new Colab notebook, choose **Runtime > Change runtime type > T4 GPU**, and run each block as its own cell:

```python
!git clone https://github.com/YOUR_GITHUB_USERNAME/Veil.git
%cd Veil
```

```python
!pip install unsloth datasets trl huggingface_hub
```

Quick smoke test (about a minute):

```python
!python train.py --data data/train.jsonl --epochs 0.1
```

Full run:

```python
!python train.py --data data/train.jsonl
```

The script prints a sample test at the end. Read it to see what the model learned.

### 6. Publish to Hugging Face

Add your token as a Colab secret named `HF_TOKEN` (key icon in the sidebar), then:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

```python
!python train.py --data data/train.jsonl --push-to-hub --hf-repo YOUR_HF_USERNAME/Veil-0.5B
```

This uploads the merged 4-bit model, plus the LoRA adapters as `Veil-0.5B-lora`. Add `--gguf` for a GGUF file you can run with llama.cpp or Ollama.

On the model page, edit the model card and start it with:

```yaml
---
license: gpl-3.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: transformers
tags: [education, test-generation, qlora, unsloth]
---
```

Describe what the model does, list the datasets you trained on, and link back to this repo.

## Make it your own

- **Subjects and grades:** the starter data is science, reading comprehension and (optionally) math. Add your own examples to `data/train.jsonl` in the same format. A few hundred good examples on your target subject helps more than thousands of generic ones.
- **Bigger or smaller model:** `python train.py --model unsloth/Qwen2.5-1.5B-Instruct ...` is better but larger. Check the base model's license.
- **Training length and memory:** use `--epochs`, `--batch-size`, `--grad-accum` and `--max-seq-len`.
- **Chat format:** `train.py` also accepts `{"messages": [...]}` lines.

## Licenses

- **Code:** GPL-3.0 (see `LICENSE`).
- **Model weights:** they depend on the base model's license and on your training data. Check both before you choose the model card license.
- **Datasets:** each dataset has its own license. As far as I know, SciQ is non-commercial (CC BY-NC) and SQuAD and ARC are share-alike (CC BY-SA), so confirm each one on its Hugging Face page. If you want a clean GPL-3.0 release, run `./get-veil-data.ps1 -Sciq 0` and credit the datasets you use.

## Limitations

Small models make mistakes. Generated questions and answer keys can be wrong, so a teacher should review every test before using it.

## Troubleshooting

- **"running scripts is disabled"** in Windows PowerShell: run `Set-ExecutionPolicy -Scope Process Bypass` once in that window, then try again.
- **`git add` says `pathspec ... did not match any files`:** one of the files you listed isn't in the folder. Use `git status` to see what exists.
- **Rate limit while downloading data:** run `./load.ps1` first so the script can use your token.
- **Colab runs out of memory:** use `--batch-size 1 --max-seq-len 1024`.
