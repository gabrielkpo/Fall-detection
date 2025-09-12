#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 22:08:07 2025
@author: gabriel

Ce script traite des dossiers contenant des vidéos décomposées en frames (images),
utilise MediaPipe pour détecter les points du squelette humain dans chaque frame,
et stocke ces données (positions des points clés) dans des tableaux .npy
pour un futur entraînement d’un modèle d’IA (ex : CNN 1D).
"""

# === Importation des bibliothèques nécessaires ===

import os                          # Pour parcourir les dossiers/fichiers du système
import numpy as np                 # Pour stocker les données efficacement et faire des calculs scientifiques
import cv2                         # Pour charger et manipuler les images (OpenCV)
import mediapipe as mp             # Pour la détection des poses humaines (MediaPipe Pose)
from tqdm import tqdm              # Pour afficher une barre de progression

# === Paramètres du traitement ===

"""
SEQUENCE_LENGTH : nombre de frames utilisées pour créer une séquence de données.
Ex : 30 frames consécutives = 1 séquence pour prédire une chute ou pas.

DATA_DIRS : dictionnaire qui associe chaque dossier à un label :
- 'fall_dataset' contient des chutes => label = 1
- 'adl_dataset' (Activities of Daily Living) contient des mouvements normaux => label = 0

IMG_SIZE : taille à laquelle toutes les images sont redimensionnées (640x480 pixels ici).
Ce redimensionnement permet une homogénéité pour la détection.
"""

SEQUENCE_LENGTH = 30                                  
DATA_DIRS = {'fall_dataset': 1, 'adl_dataset': 0}  
IMG_SIZE = (640, 480)  

# === Initialisation de MediaPipe Pose ===

"""
On utilise MediaPipe Pose en mode "static_image_mode=True" pour traiter des images indépendantes,
et non pas une vidéo en temps réel (sinon, le tracking serait activé).
Cela permet de traiter chaque frame comme une image à part entière.
"""

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# === Création des listes pour stocker les données finales ===

"""
X : contiendra des séquences de points clés (liste de listes de vecteurs de 132 valeurs).
Chaque vecteur = [x1, y1, z1, visibility1, x2, y2, z2, visibility2, ..., x33, y33, z33, visibility33]

y : contiendra les labels associés (1 pour chute, 0 pour activité normale)
"""

X = []  
y = []   

# === Fonction d'extraction des coordonnées du squelette ===

def extract_landmarks(frame):
    """
    Utilise MediaPipe pour détecter les 33 points clés du squelette humain sur une image donnée.
    Chaque point fournit 4 valeurs : x, y, z, visibility (probabilité que le point soit fiable).
    => On obtient 33 * 4 = 132 valeurs au total.

    Si aucun squelette n’est détecté, on retourne None.
    """
    # MediaPipe attend une image en RGB (OpenCV charge en BGR)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Vérifie que les points sont détectés, puis les aplatit en un seul vecteur
    if results.pose_landmarks:
        return [coord for lm in results.pose_landmarks.landmark
                for coord in (lm.x, lm.y, lm.z, lm.visibility)]
    else:
        return None

# === Parcours des dossiers de données ===

for folder, label in DATA_DIRS.items():
    print(f"Traitement de {folder}...")
    folder_path = os.path.join(os.getcwd(), folder)  # Chemin absolu du dossier à traiter
    
    """
    On liste tous les sous-dossiers valides du dossier (chacun représentant une vidéo en frames).
    Ex : fall_dataset/sequence_01/, fall_dataset/sequence_02/, etc.
    On ignore les fichiers cachés comme .DS_Store (macOS) avec os.path.isdir.
    """
    subfolders = [f for f in sorted(os.listdir(folder_path)) if os.path.isdir(os.path.join(folder_path, f))]

    for subfolder in tqdm(subfolders):  # Affiche la progression avec tqdm
        subfolder_path = os.path.join(folder_path, subfolder)
        frames = sorted(os.listdir(subfolder_path))  # Liste les images dans le bon ordre
        
        landmarks_seq = []  # Cette liste contiendra les 30 vecteurs extraits pour cette séquence

        for frame_name in frames:
            frame_path = os.path.join(subfolder_path, frame_name)
            frame = cv2.imread(frame_path)  # Charge l’image
            
            if frame is None:
                continue  # Si l’image est illisible, on la saute
            
            frame = cv2.resize(frame, IMG_SIZE)  # Redimensionne pour homogénéiser l’entrée
            
            landmarks = extract_landmarks(frame)  # Détecte les points MediaPipe
            if landmarks:
                landmarks_seq.append(landmarks)  # Ajoute le vecteur (132 valeurs)

            # Une séquence de 30 frames est atteinte => on l’enregistre
            if len(landmarks_seq) == SEQUENCE_LENGTH:
                X.append(landmarks_seq)  # Ajoute la séquence à X
                y.append(label)          # Ajoute le label correspondant (1 ou 0)
                landmarks_seq = []       # Réinitialise pour la séquence suivante

# === Sauvegarde finale des données ===

"""
X est converti en un tableau NumPy de forme (nb_séquences, 30, 132)
y est un tableau de labels (nb_séquences,)

Ces fichiers sont stockés en .npy, un format très rapide à charger avec NumPy.
Ils seront utilisés dans un futur script d'entraînement de modèle.
"""

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print(f"Enregistrement terminé : {X.shape[0]} séquences, forme = {X.shape}")
