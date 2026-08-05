# Step 1 - Collectez et structurez les données

Dans cette étape vous allez :

- Collecter, nettoyer et structurer un corpus médical bilingue (français / anglais) destiné au fine-tuning et à l’alignement par préférences.

- Produire environ 5 000 paires instruction-réponse pour SFT et constituer un jeu de paires préférentielles (DPO) validées cliniquement.

- Anonymiser toutes les données et documenter le processus RGPD.

- Définir le schéma des métadonnées (symptômes, antécédents, constantes, source, niveau de confiance).

- Préparer jeux train / val / test et jeux d’évaluations cliniques séparés.


## Prérequis

- Avoir réalisé un inventaire des sources de données disponibles (MediQA, FrenchMedMCQA, MedQuAD, UltraMedical-Preference, etc.).
- Avoir accès aux environnements de stockage et compute (espace disque, notebooks).

## Résultats attendus

- Dataset médical bilingue anonymisé et versionné, prêt pour SFT (≈5 000 paires) et pour la constitution du jeu DPO.
- Schéma des métadonnées.
- Justification du processus RGPD suivi.


## Recommandations

Vous pouvez utiliser Presidio pour anonymiser vos données. C’est un outil open source conçu pour détecter et masquer automatiquement les données sensibles.

Effectuez l’installation avec la commande :

    pip install presidio-analyzer presidio-anonymizer


Puis créez :
- un moteur d’analyse (AnalyzerEngine) pour identifier les entités sensibles, à défaut nom, prénom des patients.
- un moteur d’anonymisation (AnonymizerEngine) pour les masquer.

Pensez à utiliser un modèle linguistique adapté (ex : fr_core_news_md) et à tester différentes stratégies de masquage (replace, mask, redact) selon vos besoins.


Contrôlez la qualité du masquage pour vous assurer qu’aucune donnée personnelle identifiable ne subsiste.

- Prioriser la qualité (annotations validées) sur la quantité.

- Standardiser les formats (JSONL / HF datasets) et enregistrer les métadonnées.

- Documenter votre repository sur l’origine et la licence de chaque source ou déposer votre jeu de données sur https://huggingface.co/datasets.


## Points de vigilance

- Ne pas mélanger données d’entraînement et données d’évaluation.

- Conserver une trace de chaque transformation de données (auditabilité).


## Outils

- Hugging Face Datasets (https://huggingface.co/docs/datasets/en/index)

- Presidio (https://github.com/microsoft/presidio)
