# PAK_LEX

LLM fine-tuned on Pakistan's Constitution for legal Q&A.

## Overview
Fine-tuned LoRA adapter trained on constitutional text. Answers questions about articles, amendments, and legal provisions.

## Project Structure
```
pakistan_constitution_llm/
├── training/
│   └── final_adapter/   # LoRA adapter weights
├── venv/                # Python virtual env (ignored)
└── README.md
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```

## Model
Base model + LoRA adapter in `training/final_adapter/`.

## License
MIT
