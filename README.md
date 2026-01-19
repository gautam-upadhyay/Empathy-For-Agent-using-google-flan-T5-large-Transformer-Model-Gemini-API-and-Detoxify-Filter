# Empathy for Agents - Responsible AI Empathetic Dialogue System

An end-to-end empathy-based conversational AI that detects emotions, generates empathetic and safe responses, and serves them through a Flask backend with a modern React (Vite + Tailwind) frontend.

## Tech Stack
- Frontend: React (Vite) + TailwindCSS
- Backend: Flask (Python)
- Modeling: HuggingFace Transformers (T5 for emotion detection), Sentence-Transformers (MPNet for embeddings)
- Vector DB: ChromaDB
- Toxicity: Detoxify
- Evaluation: BLEU (sacrebleu) and ROUGE (rouge-score)
- LLM API: Gemini (google-generativeai)
- Dataset: Facebook Empathetic Dialogues

## Project Structure
```
empathy-for-agents/
  backend/
    app.py
    config.py
    preprocess.py
    tokenizer_embedder.py
    vector_db.py
    model_training.py
    response_generation.py
    validation.py
    responsible_filter.py
    requirements.txt
  frontend/
    index.html
    vite.config.js
    tailwind.config.js
    postcss.config.js
    package.json
    src/
      main.jsx
      App.jsx
      api.js
      styles.css
      components/
        ChatUI.jsx
        MessageBubble.jsx
  datasets/
    raw/
    processed/
    empathy_final.csv (generated)
  models/
    empathy_detector/
    embeddings/
    chroma_db/
```

## Quickstart

### 1) Backend setup
```bash
# From project root
cd backend
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) NLTK data for tokenization/evaluation
python - << 'PY'
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
PY

# Set env vars (PowerShell)
$env:GEMINI_API_KEY = "<YOUR_GEMINI_API_KEY>"
$env:CHROMA_DB_PATH = "../models/chroma_db"
$env:EMP_DETECTOR_PATH = "../models/empathy_detector"

# Run API
python app.py
```
Backend runs at http://localhost:5000.

### 2) Frontend setup
```bash
# From project root
cd frontend
npm install
npm run dev
```
Frontend runs at the URL Vite shows (usually http://localhost:5173).

### 3) Minimal demo (no dataset)
- The API can respond using Gemini even before training.
- For best results, run preprocessing + training first.

## Pipeline
1. Preprocess dataset → `datasets/processed/empathy_final.csv`
2. Build embeddings with MPNet → store in ChromaDB at `models/chroma_db`
3. Fine-tune T5 for emotion detection → `models/empathy_detector`
4. Serve API: detect emotion → retrieve context → generate response (Gemini) → filter toxicity (Detoxify)

## Commands

### Preprocess
```bash
cd backend
python preprocess.py --input ../datasets/raw --output ../datasets/processed/empathy_final.csv
```

### Build embeddings into ChromaDB
```bash
cd backend
python tokenizer_embedder.py --embed --input ../datasets/processed/empathy_final.csv --persist ../models/chroma_db
```

### Train T5 emotion detector
```bash
cd backend
python model_training.py --data ../datasets/processed/empathy_final.csv --out ../models/empathy_detector
```




### Performance Metric

------------- Linguistic quality/ Lexical overlap/ Is the response fluent & relevant?------------------------
## BLEU:
python -m evaluate --data ../datasets/processed/responses.csv --limit 100

## ROUGE-L: 
python -m evaluate --metric rougeL --data ../datasets/processed/responses.csv --limit 100


------------- Semantic similarity/ Does it convey similar meaning? -------------------------------------
## BERTScore
python -m evaluate --metric bertscore --data ../datasets/processed/responses.csv --limit 100


------------ EAS (Emotion Alignment Score)/ Empathy correctness/ Does it match the user’s emotion? ------------
## EAS 
python -m evaluate --metric emotion_alignment --data ../datasets/processed/responses.csv --limit 100






### Evaluate emotion detector accuracy
```bash
cd backend
python eval_emotion.py --data ../datasets/processed/empathy_final.csv --model_dir ../models/empathy_detector --limit 1000
# omit --limit to evaluate all rows (slower)
```

### Evaluate toxicity reduction
Given a CSV with a text column (default `response`), measure how many are flagged toxic before/after the responsible filter and the reduction rate:
```bash
cd backend
python eval_toxicity.py --file ../datasets/processed/responses.csv --text_col response --limit 1000
# omit --limit to check all rows
```

### Evaluate emotion alignment of generated responses
Given a CSV with generated responses and reference emotions (e.g., `response,reference_emotion`), compute alignment accuracy/F1:
```bash
cd backend
python eval_alignment.py --data ../datasets/processed/responses.csv --response_col response --label_col reference_emotion --limit 1000
# omit --limit to evaluate all rows
```

### Measure latency (p50/p90)
Benchmark `/api/chat` locally and report latency percentiles:
```bash
cd backend
python bench_latency.py --url http://127.0.0.1:5000/api/chat --message "hello" --runs 30 --concurrency 5 --use_filter
# drop --use_filter to measure without the responsible filter
```

### Evaluate (BLEU/ROUGE) on a sample CSV with columns reference, prediction
```bash
cd backend
python validation.py --file ./sample_eval.csv
```

## Environment
Set in PowerShell before running backend:
```powershell
$env:GEMINI_API_KEY = "<YOUR_KEY>"
$env:CHROMA_DB_PATH = "../models/chroma_db"
$env:EMP_DETECTOR_PATH = "../models/empathy_detector"
```

## API
- POST `/api/chat`
  - Body: `{ "message": "I failed my exam" }`
  - Response: `{ "emotion": "sadness", "response": "...", "toxic": false, "context": ["..." ] }`
- GET `/api/health` → `{ status: "ok" }`

## Notes
- Torch installation on Windows may need the official PyTorch instructions if CUDA is desired.
- If Detoxify GPU weights fail, it will fall back to CPU.
- Gemini usage requires an API key and network access.
