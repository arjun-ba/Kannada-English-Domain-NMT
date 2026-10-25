# 🌾 Neural Machine Translation for Kannada-English Domain-Specific Text

A modular, production-ready **Domain-Specific Neural Machine Translation (NMT)** system translating **Kannada (ಕನ್ನಡ) ⇄ English** in the **Agriculture and Rural Science** domain.

This project uses a **Decoupled Microservice Architecture** orchestrated by a high-performance **FastAPI Gateway**, paired with a **Zero-Dependency Interactive Web Studio** that delivers zero-latency performance without heavy frontend framework overhead.

---

## 🌟 Key Highlights & Features

- **Domain-Specific Neural Engine**:
  - Seq2Seq architecture with **Bidirectional GRU Encoder** and **Bahdanau Additive Attention Decoder**.
  - Specialized Agriculture Domain Lexicon & Terminology Extraction (e.g., *ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ* ➔ *soil testing*, *ಕೀಟಗಳ ದಾಳಿ* ➔ *pest attack*, *ಹನಿ ನೀರಾವರಿ* ➔ *drip irrigation*).
  - Neural token alignment matrix providing transparent attention weights between source and target tokens.
- **Decoupled Microservice Architecture**:
  - Changes to one service (e.g., updating tokenizer or dataset) do not break or require restarting other components.
  - Independent health monitoring and API contracts for each service.
- **Fast Interactive Web Studio**:
  - Built with Vanilla ES6+ and CSS (Glassmorphism, vibrant dark-mode theme, smooth micro-animations).
  - **Zero npm / node_modules complexity** — loads instantaneously in any browser with zero build errors.
  - Interactive Attention Matrix Heatmap table.
  - Automated BLEU Score Calculator with interactive n-gram precision bars.
  - Searchable parallel corpus explorer.
  - Native Text-to-Speech (TTS) pronunciation and one-click clipboard copying.
- **Windows UTF-8 Console Compatibility**:
  - Built-in safe stdout wrappers ensuring flawless execution in Windows PowerShell and Command Prompt without encoding crashes.

---

## 🏗️ Microservices Architecture

```
                                  [ Web Browser Studio ]
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │      FastAPI Gateway          │
                             │  (Port 8000 / Orchestrator)   │
                             └───────────────┬───────────────┘
                                             │
         ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
         ▼                   ▼                               ▼                   ▼
┌─────────────────┐ ┌─────────────────┐             ┌─────────────────┐ ┌─────────────────┐
│  Preprocessing  │ │   Translation   │             │     Dataset     │ │   Evaluation    │
│   Microservice  │ │  Microservice   │             │  Microservice   │ │  Microservice   │
├─────────────────┤ ├─────────────────┤             ├─────────────────┤ ├─────────────────┤
│ • Unicode NFC   │ │ • Seq2Seq GRU   │             │ • Parallel CSV  │ │ • BLEU-1 to 4   │
│ • Diacritics    │ │ • Attention Net │             │ • Vocab Indexer │ │ • Brevity Pen.  │
│ • Domain Lexicon│ │ • Beam / Greedy │             │ • Split & Feed  │ │ • Precision Bar │
└─────────────────┘ └─────────────────┘             └─────────────────┘ └─────────────────┘
```

1. **`services/preprocessing_service/`**:
   - `normalizer.py`: Performs Unicode NFC normalization, Kannada diacritic and zero-width non-joiner cleaning, and domain term extraction.
   - `tokenizer.py`: Handles vocabulary indexing, `<sos>`, `<eos>`, `<unk>`, and `<pad>` special tokens.
2. **`services/translation_service/`**:
   - `attention.py`: Implements Bahdanau Additive Attention tensor math.
   - `seq2seq_model.py`: Bidirectional GRU Encoder and Attention Decoder.
   - `translator.py`: End-to-end inference engine with domain fallback and attention matrix generation.
3. **`services/dataset_service/`**:
   - `data_loader.py`: Manages the parallel Kannada-English agriculture dataset, vocabularies, and train/validation splits.
4. **`services/evaluation_service/`**:
   - `evaluator.py`: Computes individual BLEU-1, BLEU-2, BLEU-3, BLEU-4, Cumulative BLEU, and Brevity Penalty (BP) with smoothing.
5. **`api_gateway/`**:
   - `server.py`: Gateway orchestrator mounting routers and static web assets.
   - `routes/`: Decoupled REST routes (`translation_routes.py`, `dataset_routes.py`, `evaluation_routes.py`, `health_routes.py`).

---

## 🚀 Quick Start Guide

### 1. Requirements & Setup
The project runs using the virtual environment in `../.venv` (or standard Python 3.10+):
```powershell
# Dependencies required: torch, fastapi, uvicorn, nltk, pandas, numpy
```

### 2. Run the Interactive Web Studio
To start the FastAPI Gateway and open the interactive Web UI:
```powershell
python run_server.py
```
- **Web Interface**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Run the Terminal CLI Demo
To test translation and evaluation directly from the terminal without a browser:
```powershell
python run_cli_demo.py
```

### 4. Run the Unit Test Suite
To verify all microservices independently:
```powershell
python -m unittest discover -s tests
```

---

## 📊 Evaluation & Metrics

The system evaluates translations using the standard **Bilingual Evaluation Understudy (BLEU)** benchmark:
$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \ln p_n\right)$$
- **BLEU-1**: Unigram precision (vocabulary adequacy).
- **BLEU-2 / BLEU-3**: Bigram & Trigram precision (domain phrase structure).
- **BLEU-4**: 4-gram precision (fluency).
- **Brevity Penalty (BP)**: Penalizes artificially short translations.
