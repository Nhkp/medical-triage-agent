# Step 0

## Définition de SFT  - Supervised Fine-Tuning

- Le Supervised Fine-Tuning est une méthode d’entraînement où l’on prend un modèle déjà pré-entraîné et on l’affine avec des exemples bien annotés (questions/réponses, consignes/réalisations, etc.).

- Le modèle apprend ainsi à mieux suivre les attentes humaines en copiant les bons comportements montrés dans ces données supervisées.


## Définition de DPO - Direct Preference Optimization

- Le Direct Preference Optimization (DPO) est une méthode d’alignement des modèles de langage basée sur des préférences humaines.
- L’idée est d’entraîner le modèle à générer des réponses qui correspondent davantage aux attentes humaines sans passer par un modèle de récompense intermédiaire.
- En pratique, le DPO permet d’apprendre directement à partir de paires de réponses annotées par des humains en indiquant laquelle est préférée.
