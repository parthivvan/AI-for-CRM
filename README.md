# GenX 360 CRM — AI-Powered Modules

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-3776AB?style=flat&logo=python)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX--Runtime-1.18+-00599C?style=flat)](https://onnxruntime.ai)
[![PyTest](https://img.shields.io/badge/Tests-20%2F20%20Passing-brightgreen?style=flat)](https://pytest.org)

An AI microservice built for the **GenX 360 CRM platform**. It provides local image-based analysis for skin/scalp/body composition, indicator-to-treatment shortlist mapping, GenAI consultant brief synthesis (Gemini 2.5 Flash), up-sell scoring engine, and consultant outcome feedback logging.

---

## 🏛 System Architecture & Scoped AI Boundaries

```
[ Client / UI ] ──────► POST /analyze (Image File)
                             │
                             ▼
               ┌───────────────────────────┐
               │   LocalModelProvider     │
               │  (ONNX Runtime / Softmax) │
               └─────────────┬─────────────┘
                             │ (Fallback Heuristics if low confidence)
                             ▼
               ┌───────────────────────────┐
               │    Indicator Mapping      │
               │  (data/treatment_rules)   │
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │     GenAI Synthesis       │
               │ (Gemini: Brief Write-up)  │
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │  Human Review & Outcomes  │
               │ (SQLite: outcome_logs)    │
               └───────────────────────────┘
```

> **Strict Architectural Boundary**: GenAI (Gemini) is strictly scoped to text synthesis (*Consultant Briefs* & *Sales Pitches*). Image classification is performed 100% locally via **ONNX Runtime** with Softmax probability normalization, falling back to deterministic rules. LLMs do **NOT** act as diagnostic visual classifiers.

---

## 📊 Honest Audit & Feature Readiness Matrix

| Module / Component | Implementation Status | Technical Reality | Grade |
| :--- | :--- | :--- | :---: |
| **Module 1: Image Analysis API** | **Complete** | `POST /analyze` runs ONNX Softmax inferencing + indicator mapping. | **A-** |
| **Module 1: Consultant UI** | **Complete** | `test_ui.html` displays image preview, Softmax probabilities, treatment shortlist, GenAI brief, safety badge & `POST /outcomes`. | **A** |
| **Module 1: ML Model Accuracy** | **Placeholder** | `models/skin_analysis.onnx` is a synthetic dummy tensor model. Fallback uses image resolution heuristics. | **C-** |
| **Module 2: Upsell Scoring API** | **Complete** | `POST /recommendations/run` calculates LTV, visit interval & package scores + GenAI pitch scripts. | **B+** |
| **Module 2: Staff UI** | **Pending** | No HTML dashboard UI built for Module 2 staff list (API endpoint ready). | **F** |
| **Database & Audit Trail** | **Complete** | SQLite (`genx_ai.db`) via SQLAlchemy storing `analysis_logs`, `recommendation_logs`, and `outcome_logs`. | **A** |
| **GenAI Governance** | **Complete** | Gemini 2.5 Flash strictly restricted to text write-ups. Vision fallback is 100% deterministic. | **A+** |

---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.11+ or 3.13+
* Git

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/parthivvan/AI-for-CRM.git
cd AI-for-CRM

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies in editable mode with test tools
pip install -e ".[test]"

# Configure environment variables
copy .env.example .env
```

### 3. Run the Service

```powershell
# Launch FastAPI development server with auto-reload
uvicorn app.main:app --reload
```

* **Interactive OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Module 1 Consultation UI**: [http://127.0.0.1:8000/test-ui](http://127.0.0.1:8000/test-ui)

---

## 🔌 API Endpoints Reference

### 1. Image Analysis & Shortlist (`POST /analyze`)
Accepts image upload, runs local ONNX classification (Softmax probabilities), maps indicators to treatment shortlist, and generates a GenAI Consultant Brief.

* **Headers**: `X-API-Key: test-api-key` (optional in dev)
* **Form Parameters**:
  * `file`: Image file (`.jpg`, `.png`)
  * `analysis_type`: `skin` | `scalp` | `body` (default: `skin`)
  * `branch_id`: Branch ID (e.g., `B001`)
  * `client_id`: Client ID (e.g., `C101`)

### 2. Consultant Outcome Feedback (`POST /outcomes`)
Logs consultant human-in-the-loop decisions (accept, edit, or reject treatment recommendations) for safety compliance and model retraining audit.

* **JSON Payload**:
  ```json
  {
    "analysis_id": 1,
    "consultant_id": "EMP_102",
    "decision": "accept",
    "notes": "Confirmed pigmentation on cheeks."
  }
  ```

### 3. Cross-Sell / Up-Sell Next-Best-Action (`POST /recommendations/run`)
Generates ranked up-sell recommendations based on client $LTV$, visit recency, and preferred categories, along with a personalized front-desk sales pitch.

* **JSON Payload**:
  ```json
  {
    "client_id": "C101",
    "branch_id": "B001",
    "min_score": 40
  }
  ```

---

## 🧪 Testing & Verification

Run the full automated test suite (20 tests covering ONNX tensor inferencing, Softmax sum-to-1 assertion, fallback vision rules, and outcome feedback logging):

```powershell
pytest
```

---

## 📂 Project Structure

```
AI-for-CRM/
├── app/
│   ├── main.py                  # FastAPI app & endpoint routing
│   ├── ai_provider.py           # GenAI brief synthesis & classification fallback
│   ├── local_model_provider.py  # Local ONNX model loader & Softmax activation
│   ├── rules.py                 # Indicator-to-treatment & up-sell scoring engine
│   ├── storage.py               # SQLAlchemy SQLite ORM models & database session
│   └── security.py              # API key verification middleware
├── data/
│   ├── treatment_rules.json     # Treatment routing keyword catalog
│   └── sample_crm.json          # Mock CRM client records
├── models/
│   └── skin_analysis.onnx       # Local ONNX classification model
├── tests/
│   ├── test_module1_flow.py     # Integration test for analysis & outcomes flow
│   ├── test_local_model_provider.py # ONNX Softmax unit tests
│   ├── test_provider_fallback.py    # Fallback vision routing tests
│   └── test_rules.py            # Business logic unit tests
├── test_ui.html                 # Consultant UI with Human Review safety badge
├── GenX_360_CRM_Documentation.docx # Comprehensive technical audit document
├── pyproject.toml               # Package configuration & dependencies
└── README.md                    # Project documentation
```

---

## 📄 License & Audit Status

* **Audit Document**: Detailed technical audit report is saved as [GenX_360_CRM_Documentation.docx](GenX_360_CRM_Documentation.docx).
* **Current Status**: All 20 tests passing (100%). Ready for production model weight replacement and live CRM API hookup.
