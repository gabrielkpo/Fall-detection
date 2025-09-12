# Détection Automatique de Chutes

Une application Python basée sur l’IA pour détecter automatiquement les chutes à partir de vidéos.  
Elle utilise **MediaPipe** pour extraire les coordonnées articulaires et un **réseau de neurones convolutif 1D (CNN)** pour analyser les séquences temporelles et identifier les chutes en temps réel.  
Un système d’alerte (sonore et notifications) permet une intervention rapide et efficace.

---

## Sommaire
- [Fonctionnalités](#fonctionnalités)  
- [Installation](#installation)  
- [Utilisation](#utilisation)  
- [Organisation du projet](#organisation-du-projet)  
- [Modèle et entraînement](#modèle-et-entraînement)  
- [Système d’alerte](#système-dalerte)  
- [Résultats et métriques](#résultats-et-métriques)  
- [Axes d’amélioration](#axes-damélioration)  
- [Dépendances](#dépendances)  
- [FAQ](#faq)  
- [Auteur](#auteur)  

---

## Fonctionnalités
- Extraction des coordonnées articulaires 3D via **MediaPipe**  
- Segmentation des vidéos en séquences temporelles (2-3s)  
- Modèle **CNN 1D** entraîné pour classification binaire (chute / non-chute)  
- Option alternative avec **autoencodeur** pour détection d’anomalies  
- Détection en temps réel avec capture d’images clés  
- **Alerte sonore** en cas de chute détectée  
- Enregistrement des événements (logs + vidéos)  
- Possibilité d’envoyer des **notifications**  
- Gestion de la variabilité des angles de caméra via :  
  - Caractéristiques invariantes (distances relatives, angles articulaires)  
  - Augmentation artificielle des données par rotation  

---

## Installation
1. Assurez-vous d’avoir **Python 3.8+** installé.  
2. Clonez le dépôt :  
   ```bash
   git clone https://github.com/ton-profil/DetectionChutes.git
   cd DetectionChutes
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt

---

## Utilisation

Exécution en temps réel (webcam)
```bash
python scripts/run_realtime.py
```

Entraînement du modèle
```bash
python scripts/train.py --data data/processed --epochs 50
```

Prédiction sur une vidéo
```bash
python scripts/predict.py --video data/videos/test_video.mp4
```

Les événements détectés (images, vidéos et logs) sont sauvegardés dans data/events/.

---

## Organisation du projet

- **src/**: Modules Python (modèles, prétraitement, utilitaires)

- **scripts/** : Scripts d’exécution (entraînement, prédiction, temps réel)

- **data/raw/** : Données brutes (vidéos originales)

- **data/processed/** : Données transformées (séquences, features)

- **data/videos/** : Vidéos de test

- **data/events/** : Enregistrements de chutes détectées

- **models/** : Modèles sauvegardés

- **notebooks/** : Expérimentations et analyses Jupyter

---

## Modèle et entraînement

**Architecture CNN 1D** :

- 3 couches convolutives

- 32 filtres, kernel size = 5

- AdaptiveMaxPool1d en sortie

**Fonction de perte** : Binary Cross-Entropy

**Métriques** : Accuracy, Precision, Recall, F1-score

**Stratégies anti-surapprentissage** :

- Dropout / Weight Decay

- Early Stopping

- Augmentation de données

---

## Système d’alerte

- Détection en temps réel via webcam

- Alerte sonore immédiate en cas de chute

- Sauvegarde vidéo + image clé pour analyse ultérieure

- Notifications configurables pour surveillance à distance

---

## Résultats et métriques

Précision globale satisfaisante (>90% sur données de validation)

Bonne capacité de généralisation aux mouvements variés

Quelques limites observées pour les changements extrêmes d’angle de caméra

---

## Axes d’amélioration

Intégration de vidéos multi-caméras

Hybridation CNN + LSTM pour capter les dépendances temporelles longues

Enrichissement du dataset par des scénarios plus variés

Optimisation du modèle pour déploiement sur appareils embarqués (Raspberry Pi, IoT)

---

## Dépendances

Python 3.8+

MediaPipe

OpenCV

NumPy / Pandas

PyTorch / TensorFlow (selon implémentation)

Matplotlib / Seaborn (visualisation)

---

## FAQ

Q : **Peut-on utiliser ce projet avec une webcam classique ?**
R : Oui, toute webcam standard est supportée via OpenCV.

Q : **Le modèle fonctionne-t-il hors ligne ?**
R : Oui, tout est embarqué localement.

Q : **Comment ajouter mes propres vidéos d’entraînement ?**
R : Placez-les dans data/raw/ puis utilisez scripts/preprocess.py pour les transformer.

Q : **Peut-on détecter d’autres événements que les chutes ?**
R : Oui, en enrichissant le dataset et en ajustant les labels, le modèle peut être adapté à d’autres gestes.

---

## Auteur

Développé par Kpodoh Gabriel — 2025
