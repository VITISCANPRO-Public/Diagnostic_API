<<<<<<< HEAD
# VitiScan Diagno API

📌 Description

Ce projet expose une API permettant d’analyser des feuilles de vigne afin de :

- Diagnostiquer une maladie à partir d’une image.
- Prédire réellement la maladie avec un modèle CNN.

L’API est construite avec FastAPI et peut être déployée sur Hugging Face Spaces ou intégrée avec MLFlow pour le suivi des modèles.

📁 Structure du projet

vitiscan-diagno-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # API FastAPI (endpoints /diagno et /predict)
│   ├── model.py         # Fonctions de chargement et prédiction CNN
│   ├── schemas.py       # Schémas Pydantic pour les réponses JSON
│   └── utils.py         # Fonctions utilitaires
├── models/              # Modèle CNN exporté
├── tests/
│   └── test_predict.py  # Tests unitaires pour /diagno et /predict
├── requirements.txt     # Dépendances Python
├── Dockerfile           # Déploiement Hugging Face Space
├── .env                 # Variables d’environnement
└── README.md
=======
---
title: Vitiscan Diagno Api
emoji: 🚀
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> hf_inti/main
