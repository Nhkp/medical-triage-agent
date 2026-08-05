# Step 3 - Déployez et validez le POC

- Automatiser le déploiement du prototype via le pipeline CI/CD mis en place sur GitHub Actions.
- Conteneuriser l'application avec Docker et l'exposer via une API (FastAPI) en utilisant vLLM pour une inférence optimisée.
- Réaliser des tests de latence, de robustesse, ainsi que des audits de traçabilité des interactions.

## Prérequis

- Le modèle fine-tuné et validé est disponible.
- Le pipeline CI/CD sur GitHub est fonctionnel.


## Résultats attendus

- Un endpoint de démonstration déployé et accessible en environnement pilote.
- Un processus de déploiement automatisé et reproductible.
- Le rapport final incluant les métriques de performance et la roadmap de déploiement.


## Recommandations

- Mesurer la latence et le temps de réponse en conditions réalistes.
- Préparer un plan de mise en production conditionnel (checklist « go / no-go »).


## Points de vigilance

- Protéger les clés / secrets et l’accès aux endpoints.
- Prévoir procédures de surveillance après déploiement.
- Documenter clairement les limites d’usage pour les utilisateurs.


## Outils

- vLLM / Docker / FastAPI.
- Outils de CI/CD et tests d’intégration (GitHub Actions, GitLab CI).


## Ressource

Documentation vLLM & exemples de déploiement (https://docs.vllm.ai/en/latest/)
