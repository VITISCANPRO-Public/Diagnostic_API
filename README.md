---
title: VITISCANPRO_DIAGNO_API
emoji: 🍇
colorFrom: green
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Vitiscan — Diagnostic API

REST API for grape leaf disease classification using a fine-tuned CNN model.
Part of the **Vitiscan MLOps pipeline**.

## Overview

Accepts a grape leaf image and returns a ranked list of disease predictions
with confidence scores, using a ResNet18 model served via MLflow.

**Endpoint:** `POST /diagno`  
**Classes:** Colomerus vitis, Elsinoe ampelina, Erysiphe necator, Guignardia bidwellii,
Phaeomoniella chlamydospora, Plasmopara viticola, Healthy

## Project Structure
```
Diagnostic-API/
├── app/
│   ├── main.py       # FastAPI app and endpoints
│   ├── model.py      # Model loading and inference
│   ├── schemas.py    # Pydantic request/response schemas
│   └── utils.py      # Helper functions
├── tests/
│   └── test_predict.py
├── Dockerfile
└── requirements.txt
```

## Quickstart

**1. Setup environment**
```bash
pip install -r requirements.txt
```

**2. Configure your `.env`**
```bash
cp .env.example .env
# Fill in MLFLOW_TRACKING_URI, MLFLOW_MODEL_URI, AWS credentials
```

**3. Run locally**
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 4000 --reload
```

API docs available at `http://127.0.0.1:4000/docs`

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/diseases` | List detectable diseases |
| POST | `/diagno` | Run diagnosis on uploaded image |

## Deployment

Deployed on HuggingFace Spaces (Docker) at:  
`https://mouniat-vitiscanpro-diagno.hf.space`

## Requirements

- Python 3.11
- MLflow 3.7.0
- PyTorch 2.x
- See `requirements.txt` for full list