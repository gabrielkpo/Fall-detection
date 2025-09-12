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

### Organisation du projet

**src/**: Modules Python (modèles, prétraitement, utilitaires)

scripts/ : Scripts d’exécution (entraînement, prédiction, temps réel)

data/raw/ : Données brutes (vidéos originales)

data/processed/ : Données transformées (séquences, features)

data/videos/ : Vidéos de test

data/events/ : Enregistrements de chutes détectées

models/ : Modèles sauvegardés

notebooks/ : Expérimentations et analyses Jupyter
