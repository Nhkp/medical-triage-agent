# Context

## Introduction

Vous êtes missionné en tant qu'IA Engineer junior pour le compte du Centre Hospitalier Saint-Aurélien (CHSA).

Le CHSA est un grand hôpital public français dont le service des urgences connaît une surcharge constante, particulièrement aux heures de pointe. Le personnel infirmier et de triage manque parfois d'effectifs, entraînant des temps d'attente prolongés et un risque de négligence des cas critiques non identifiés rapidement.


Vous êtes chargé de développer un POC (Proof of Concept) d'un agent IA de triage médical en 4 semaines.


## Mail

Bonjour,

Comme convenu lors de notre entretien, je reviens vers vous pour préciser le contexte et les attendus de la mission confiée par la Direction du Centre Hospitalier Saint-Aurélien.

Le CHSA souhaite, face à la surcharge constante de son service d'urgences, disposer d'un agent intelligent permettant d'assister le personnel soignant dans le triage initial des patients.


L'agent IA devra accompagner les patients en :
- Collectant leurs symptômes via un questionnaire intelligent adaptatif,
- Évaluant le niveau de priorité (urgence maximale / modérée / différée) selon les protocoles médicaux,
- Fournissant des explications claires sur l'évaluation et les recommandations,
- S'intégrant au système d'information hospitalier existant,
- Garantissant la traçabilité de chaque interaction pour les audits médicaux.

Notre rôle est de réaliser un Proof of Concept (POC) qui démontre la faisabilité technique et la valeur ajoutée clinique d'un tel système.

### Approche Technique et Stratégie Expérimentale
Les avancées récentes en intelligence artificielle médicale démontrent que les modèles de langage spécialisés peuvent atteindre des performances diagnostiques comparables à celles de médecins en formation. N’hésitez pas à consulter cet article. Toutefois, leur déploiement en environnement clinique exige une méthodologie rigoureuse et une validation approfondie.

#### Notre stratégie s'articule autour de trois phases progressives :

- ##### Phase 1 - Validation Conceptuelle :
  Nous déploierons Qwen3-1.7B-Base, un modèle compact mais performant, permettant de valider rapidement nos hypothèses techniques tout en évaluant l'acceptabilité clinique du système par les équipes soignantes.
- ##### Phase 2 - Optimisation Ciblée :
  Le modèle sera affiné par fine-tuning supervisé (SFT) avec la technique LoRA, puis optimisé via l'alignement par préférences via DPO pour garantir sa conformité aux protocoles médicaux établis.
- ##### Phase 3 - Projection Industrielle :
  En cas de validation concluante du POC, nous envisagerons le passage à des modèles de plus grande envergure (32B+ paramètres) avec des jeux de données étendus pour la mise en production. L'architecture des données médicales (symptomatologie, antécédents, constantes vitales, protocoles de triage) sera particulièrement soignée pour optimiser l'apprentissage du modèle.

### Feuille de route et mission détaillée
Pour respecter notre échéance de 4 semaines, vos missions s'organisent selon le planning suivant :

#### Semaine 1 - Préparation et structuration des données
- Agrégation des corpus médicaux francophones et anglophones
  - MediQA, (https://huggingface.co/datasets/ANR-MALADES/MediQAl)
  - FrenchMedMCQA, (http://nthngdy/frenchmedmcqa)
  - MedQuAD, (http://keivalya/MedQuad-MedicalQnADataset)
  - UltraMedical-Preference. (http://keivalya/MedQuad-MedicalQnADataset)

- Constitution d'un dataset d'entraînement SFT de 5 000 paires instruction-réponse, optimisé pour le fine-tuning.

- Création du dataset de post-entraînement DPO avec paires de réponses validées/non validées.

- Anonymisation et validation de la conformité RGPD des données.

#### Semaine 2 - Entraînement Initial par Fine-Tuning Supervisé (SFT)

Au cours de cette semaine, l'objectif principal est de spécialiser le modèle de base sur notre corpus médical.

- Implémentation du SFT : Nous lancerons le fine-tuning supervisé du modèle Qwen3-1.7B-Base.
- Optimisation par LoRA : La méthode LoRA (Low-Rank Adaptation) sera utilisée pour adapter le modèle de manière efficace, en optimisant l'usage des ressources GPU disponibles.
- Validation intermédiaire : Des premières évaluations seront menées sur un jeu de données de test pour mesurer les progrès du modèle et s'assurer que l'entraînement se déroule correctement.

#### Semaine 3 - Alignement du Modèle par Préférences (DPO)

Cette semaine est dédiée à l'affinage du comportement du modèle pour qu'il corresponde aux attentes et aux pratiques cliniques.

- Entraînement DPO : Le modèle préalablement fine-tuné sera entraîné avec la méthode DPO (Direct Preference Optimization). Cet entraînement se basera sur les paires de réponses préférentielles du dataset UltraMedical-Preference.

- Alignement clinique : L'objectif de cette phase est d'aligner plus finement les réponses du modèle sur les pratiques cliniques validées, en lui apprenant à distinguer les réponses de meilleure qualité des réponses moins pertinentes ou incorrectes.

#### Semaine 4 - Déploiement et validation

- Mise en production pilote
  - Déploiement d'un endpoint prototype via vLLM.
  - Simulation d'inférence en conditions quasi-réelles.
  - Tests de latence, pertinence et traçabilité des interactions.

- Évaluation finale
  - Analyse des métriques de performance.
  - Rédaction du rapport de synthèse.
  - Formulation des recommandations stratégiques pour le passage à l'échelle.


### Livrables attendus
À l'issue de cette mission de 4 semaines, vous devrez fournir :

- Dataset médical bilingue prêt à l'emploi : Un corpus de données médicales (francophone et anglophone) entièrement nettoyé, structuré et anonymisé en conformité avec le RGPD. Ce dataset sera optimisé pour les phases de fine-tuning (SFT) et d'alignement (DPO).

- Modèle d'IA spécialisé et optimisé : Le modèle de langage Qwen3-1.7B, fine-tuné avec les techniques SFT et LoRA, puis aligné par préférences (DPO) pour garantir la pertinence clinique de ses réponses. Les poids finaux du modèle seront fournis.

- Endpoint de démonstration déployé sur le cloud : Une interface de démonstration fonctionnelle et accessible via une API. L'endpoint sera déployé sur le fournisseur de cloud de votre choix et optimisé pour une inférence rapide grâce à la technologie vLLM.

- Pipeline d'intégration et de déploiement continu (CI/CD) : Un pipeline CI/CD mis en place avec GitHub Actions. Ce pipeline automatisera les tests et le déploiement de nouvelles versions du modèle, assurant ainsi la maintenabilité et l'évolutivité de la solution.

- Rapport technique complet et recommandations stratégiques : Un document de 20 pages maximum détaillant :
  - La méthodologie employée pour la préparation des données et l'entraînement du modèle.
  - Les métriques de performance (latence, pertinence, etc.).
  - Une analyse des résultats obtenus.
  - Une roadmap claire pour le passage à l'échelle et le déploiement à plus grande échelle au sein du CHSA.


Cette mission représente un enjeu stratégique majeur pour l'amélioration de la prise en charge des patients au CHSA. Nous comptons sur votre expertise technique et votre rigueur méthodologique pour concrétiser cette innovation au service de la santé publique.


Merci pour votre investissement.
Bien cordialement


Dr. Marie Dubois
