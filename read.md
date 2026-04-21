# 🫁 Détection automatique de la pneumonie à partir de radiographies pulmonaires (Deep Learning)

## 📌 Contexte

La pneumonie est une infection pulmonaire fréquente et potentiellement grave, nécessitant un diagnostic rapide et fiable. L’analyse des radiographies thoraciques (Chest X-ray) est une étape clé, mais elle reste dépendante de l’expertise médicale et peut être sujette à variabilité.

Dans ce contexte, l’intelligence artificielle, et en particulier le **deep learning**, offre des perspectives prometteuses pour assister le diagnostic.

---

## 🎯 Objectif du projet

Ce projet vise à concevoir un modèle de deep learning capable de :

* Classifier automatiquement des radiographies pulmonaires en :

  * **Normal**
  * **Pneumonie**
* Exploiter le **transfer learning** pour améliorer les performances
* Mettre en place un pipeline complet :

  > données → prétraitement → modèle → évaluation → déploiement
* Proposer une **interface utilisateur simple (Flask)** pour tester le modèle

---

## 📂 Structure du projet
```text
Pneumonia_Detection/
├── app/                      # Application Flask (interface utilisateur)
├── data/                     # Dataset (non inclus sur GitHub)
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
├── models/                   # Modèles entraînés (.h5)
├── notebooks/                # Notebook principal du projet
│   └── Pneumonia_Detection_CNN.ipynb
├── Demo_images/                   # Images de démonstration
├── results/                  # Figures (courbes, résultats)
├── README.md
├── requirements.txt
└── .gitignore
```
---
## 🚀 Installation

1. Cloner le projet
git clone https://github.com/salimfourati/pneumonia-detection.git
cd pneumonia-detection
2. Installer les dépendances
pip install -r requirements.txt

---
## 📊 Données

Dataset utilisé : **Chest X-Ray Pneumonia (Kaggle)**

* Images radiographiques pulmonaires
* Deux classes :

  * Normal
  * Pneumonie
* Séparation :

  * Entraînement
  * Validation
  * Test

⚠️ Le dataset n’est pas inclus dans le dépôt.

📥 Téléchargement :
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Structure attendue :

```text
data/chest_xray/
├── train/
├── val/
└── test/
```

---

## ⚙️ Pipeline méthodologique

### 🔹 1. Prétraitement

* Redimensionnement des images
* Normalisation
* Data augmentation :

  * rotation
  * zoom
  * retournement horizontal

👉 Objectif : améliorer la robustesse et limiter l’overfitting.

---

### 🔹 2. Modélisation

Le modèle repose sur :

* **VGG16 pré-entraîné sur ImageNet**
* Réutilisation des couches convolutionnelles (extraction de features)
* Ajout d’une **tête de classification personnalisée**

---

### 🔹 3. Stratégie d’entraînement

Deux phases distinctes :

#### Phase 1 – Transfer Learning

* Gel des couches de VGG16
* Entraînement uniquement des couches finales

#### Phase 2 – Fine-tuning

* Dégel progressif de certaines couches profondes
* Ajustement des poids pour le domaine médical

👉 Cette stratégie permet :

* d’exploiter des features génériques
* tout en les adaptant aux radiographies

---

### 🔹 4. Optimisation

* Optimiseur : **Adam**
* Fonction de perte : **Binary Crossentropy**
* Suivi des performances sur validation

---

## 📈 Résultats

Le modèle montre une bonne capacité à distinguer les deux classes.

### Métriques utilisées :

* Accuracy
* Precision
* Recall (**critique en contexte médical**)
* F1-score

### Analyse qualitative :

* Bonne détection des cas de pneumonie
* Sensibilité élevée souhaitable (réduction des faux négatifs)
* Quelques confusions possibles dues :

  * à la qualité des images
  * au déséquilibre du dataset

📊 Les courbes d’apprentissage sont disponibles dans `results/`.

---

## 🖼️ Prédictions

Le modèle est capable de prédire sur de nouvelles images :

* Chargement d’une radiographie
* Passage dans le modèle
* Classification automatique


---

## 🌐 Application Flask

Une application web permet d’utiliser le modèle de manière interactive.

### Fonctionnalités :

* Upload d’une image
* Prédiction en temps réel
* Affichage du résultat

### Lancement :

```bash
cd app
pip install -r requirements.txt
python app.py
```

Accès :

```text
http://127.0.0.1:5000
```

---

## ⚠️ Limites du projet

* Dataset potentiellement déséquilibré
* Données issues d’une seule source (généralisation limitée)
* Absence d’explicabilité (black-box)
* Sensibilité au bruit et à la qualité des images

---

## 🔧 Améliorations possibles

* Intégration de **Grad-CAM** (explicabilité)
* Utilisation de modèles plus récents :

  * ResNet
  * EfficientNet
* Optimisation des hyperparamètres
* Augmentation avancée des données
* Déploiement cloud (API ou application web complète)

---

## 🧠 Discussion

Ce projet illustre l’intérêt du **transfer learning en imagerie médicale**, notamment lorsque les données sont limitées.

Le recours à un modèle pré-entraîné permet de :

* réduire le temps d’entraînement
* améliorer la convergence
* obtenir des performances robustes

Cependant, la généralisation reste un enjeu majeur en contexte clinique réel.

---

## 👨‍💻 Auteur

**Salim Fourati**
Ingénieur en Génie Biomédical
Master IA – Vision & Robotique

---

## 📄 Licence

Projet réalisé dans un cadre académique et pédagogique.
