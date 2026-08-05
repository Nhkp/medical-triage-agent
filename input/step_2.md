# Step 2 - Affinez et alignez le modèle

- Effectuer le fine-tuning supervisé (SFT) du modèle Qwen3-1.7B-Base en utilisant LoRA afin de limiter l’empreinte GPU, puis entraîner l’alignement par préférences (DPO / GRPO) sur les paires préférentielles.
- Valider les performances intermédiaires sur les jeux de test cliniques et réaliser les contrôles de sécurité (hallucinations, recommandations dangereuses).
- Itérer sur les hyperparamètres et les checkpoints en gardant une traçabilité (fichiers de logs, métriques et checkpoints pour la reprise de l'entraînement si nécessaire).

## Prérequis

- Avoir accès à GPUs et à un environnement ML (PyTorch / HF transformers / Unsloth ).
- Avoir défini les métriques d’évaluation cliniques et les seuils d’acceptation.
- Avoir mis en place la sauvegarde/monitoring des modèles et des logs d’entraînement.

## Résultat attendu

- Modèle Qwen3-1.7B  adapté (SFT LoRA + DPO) avec métriques d’évaluation documentées et checkpoints reproductibles.


## Recommandations

- Commencer par des petits runs LoRA pour valider la pipeline avant la montée en charge.
- Mettre en place des checkpoints.
- Enregistrer et documenter chaque version (hyperparamètres + seed).


## Points de vigilance

- Éviter le sur-apprentissage sur les exemples annotés.


## Outils

- PyTorch / Hugging Face Transformers / PEFT (LoRA).
- MLflow ou Weights & Biases pour tracking.


## Ressources

- Tutoriels LoRA + PEFT [nom & lien]. (https://docs.unsloth.ai/get-started/unsloth-notebooks)
- Exemple de notebook LoRA + SFT (https://docs.unsloth.ai/get-started/unsloth-notebooks)
