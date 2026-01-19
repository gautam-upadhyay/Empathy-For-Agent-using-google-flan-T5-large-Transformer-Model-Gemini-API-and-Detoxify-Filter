# Empathy for Agents - Responsible AI Empathetic Dialogue System

An end-to-end empathy-based conversational AI that detects emotions, generates empathetic and safe responses, and serves them through a Flask backend with a modern React (Vite + Tailwind) frontend.

## Tech Stack
- Frontend: React (Vite) + TailwindCSS
- Backend: Flask (Python)
- Modeling: HuggingFace Transformers (T5 for emotion detection)
- Embeddings: Sentence-Transformers (MPNet)
- Vector DB: ChromaDB
- Toxicity: Detoxify
- Evaluation: BLEU (sacrebleu), ROUGE-L (rouge-score), BERTScore, Emotion Alignment (accuracy/F1)
- LLM API: Gemini (google-generativeai)
- Dataset: EmpatheticDialogues (train/valid/test)

## Project Structure
```
Empathy_For_Agent/
  backend/
    app.py
    cli_chat.py
    config.py
    evaluate.py
    model_training.py
    preprocess.py
    response_generation.py
    responsible_filter.py
    tokenizer_embedder.py
    user_memory.py
    vector_db.py
    requirements.txt
    scripts/
      fix_typos.py
      query_chroma.py
  frontend/
    index.html
    app.js
    styles.css
    vite.config.js
    tailwind.config.js
    postcss.config.js
    package.json
    src/
      main.jsx
      App.jsx
      api.js
      index.css
      components/
        ChatArea.jsx
        ChatUI.jsx
        MessageBubble.jsx
        Sidebar.jsx
  datasets/
    raw/
      empatheticdialogues/
        train.csv
        valid.csv
        test.csv
    processed/
      empathy_final.csv
      responses.csv
  models/
    empathy_detector/
    chroma_db/
  BackUp4me/
    generate_documentation.py
```

## Quickstart

### 1) Backend setup
```bash
# From project root
cd backend
python -m venv .venv
# Windows PowerShell (from backend/)
& .\.venv\Scripts\Activate.ps1
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




### Performance Metrics (evaluate.py)

Linguistic quality / lexical overlap (response fluency & relevance):

**BLEU**
```bash
cd backend
python evaluate.py --metric bleu --data ../datasets/processed/responses.csv --limit 1000
```

**ROUGE-L**
```bash
cd backend
python evaluate.py --metric rougeL --data ../datasets/processed/responses.csv --limit 1000
```

Semantic similarity (meaning overlap):

**BERTScore**
```bash
cd backend
python evaluate.py --metric bertscore --data ../datasets/processed/responses.csv --limit 1000
```

Emotion Alignment Score (EAS) — empathy correctness:

**Emotion alignment**
```bash
cd backend
python evaluate.py --metric emotion_alignment --data ../datasets/processed/responses.csv --limit 1000
```






### Evaluate emotion detector accuracy
Use the emotion alignment metric with a CSV that contains **text** and **reference_emotion** columns.
```bash
cd backend
python evaluate.py --metric emotion_alignment --data ../datasets/processed/responses.csv --limit 1000
# omit --limit to evaluate all rows (slower)
```

### Evaluate toxicity reduction
Given a CSV with **text** and **response** columns, measure toxicity before/after the responsible filter:
```bash
cd backend
python evaluate.py --metric toxicity --data ../datasets/processed/responses.csv --limit 1000
# add --use_filter to apply the responsible filter
```

### Evaluate emotion alignment of generated responses
Given a CSV with **text** and **reference_emotion** columns, compute alignment accuracy/F1:
```bash
cd backend
python evaluate.py --metric emotion_alignment --data ../datasets/processed/responses.csv --limit 1000
# omit --limit to evaluate all rows
```

### CLI chat (optional)
Run a local terminal chat loop:
```bash
cd backend
python cli_chat.py
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
