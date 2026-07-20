# AI-for-CRM: GenX 360 AI Service

This repository contains the pilot AI microservice for the GenX 360 CRM. It provides image-based analysis for skin/scalp conditions, AI-assisted treatment recommendations, and outcome tracking. The service acts as a standalone REST API that integrates with the primary Node/Express CRM.

---

## 📊 Project Status: What's Done vs. What's Left

**Overall Progress: ~85% Complete**

### ✅ Completed (Done)

**1. Core Backend & Architecture**
- **FastAPI Server**: Setup with robust asynchronous routing (`app/main.py`).
- **Security**: Middleware requiring `X-API-Key` validation (`app/security.py`).
- **Database**: SQLAlchemy storage engine implemented for tracking consultant outcomes via SQLite (local) or PostgreSQL (production) (`app/storage.py`).

**2. AI & ML Integration**
- **Local Vision Model (ONNX)**: `LocalModelProvider` built to load and execute ONNX computer vision models on CPU, featuring ImageNet normalisation, multi-label classification, and dynamic confidence scoring (`app/local_model_provider.py`).
- **LLM Integration**: Google Gemini API integrated via `AIProvider` to generate natural, human-readable explanations for recommended treatments (`app/ai_provider.py`).
- **Graceful Fallbacks**: The system elegantly falls back to deterministic, rule-based text if Gemini fails or hits rate limits.

**3. Business Logic & Rules**
- **Routing Engine**: Rule-based ranking engine mapping detected visual flags to specific treatment keywords (`app/rules.py`).
- **Configuration**: Hardcoded JSON configuration for clinical treatment mappings (`data/treatment_rules.json`).

**4. Data Preparation & Training Pipeline**
- **MLOps**: Comprehensive training, evaluation, and ONNX export scripts are ready for continuous fine-tuning (`training/train.py`, `evaluate.py`, `export_onnx.py`).

**5. Testing & Infrastructure**
- **Testing**: Test suite implemented covering API contracts, business rules, LLM fallback logic, and local models (`tests/`).
- **DevOps**: Docker containerization (`Dockerfile`) and Infrastructure-as-Code (`render.yaml`) are fully configured.

---

### 🚧 Pending (Left to be Done)

**1. Final ONNX Model Placement**
- The fully trained `mobilenetv3_large_skin_ai.onnx` file and its accompanying `labels.json` need to be moved into the `models/current/` directory to enable local visual analysis.

**2. Live CRM Network Integration**
- The service currently uses fallback mock data when `CRM_BASE_URL` is empty. The main GenX CRM team needs to deploy their side of the API contract (`/ai/branches/{id}/treatments`, etc.) to establish the live data link.

**3. Production Deployment & Env Config**
- Provision a production PostgreSQL database.
- Deploy the Docker container to Render/Railway and inject real environment variables (`GEMINI_API_KEY`, `CRM_API_KEY`, etc.).

**4. Automated Scheduler Activation**
- The background `scheduler.py` module is built but deliberately disabled (`ENABLE_SCHEDULER=false`). It must be manually validated by consultants (Phase 1) before being activated for automated bulk upsell processing (Phase 2).

---

## 🚀 Local Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -e ".[test]"

# 3. Setup environment variables
copy .env.example .env

# 4. Run the server
uvicorn app.main:app --reload
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health and environment status |
| `POST` | `/analyze` | Run image analysis & get treatment recommendations |
| `POST` | `/recommendations/run` | Bulk trigger recommendations for a branch |
| `POST` | `/outcomes` | Log consultant feedback (accepted/rejected) |

*Note: All `POST` endpoints require the header `X-API-Key: <your-key>`.*
