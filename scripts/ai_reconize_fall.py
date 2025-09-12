#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 21:56:57 2025

@author: gabriel

Pour exécuter le script dans le terminal : 
 
cd ~/Desktop/Perso/projet_info/Reco_mouv/cnn_fall
python3 ai_reconize_fall.py


"""

# %%

""" 

Importe la bibliothèque OpenCV pour la capture vidéo et le traitement d’image 
Importe MediaPipe, une bibliothèque Google pour la détection de pose, visage, etc. 

""" 

import cv2
import mediapipe as mp

""" 
Initialise le module de détection de pose humaine de MediaPipe 
"""

mp_pose = mp.solutions.pose

""" 
Initialise les fonctions de dessin de MediaPipe (pour afficher les points détectés sur l'image) 
"""

mp_drawing = mp.solutions.drawing_utils

""" 
Ouvre la webcam locale (ID 0 correspond à la webcam principale de l’ordinateur) 
"""

cap = cv2.VideoCapture(0)

""" 
Crée un contexte d’utilisation du détecteur de pose avec les paramètres choisis 
"""

with mp_pose.Pose(
    static_image_mode=False,  # False = détection en continu (stream vidéo)
    model_complexity=1,       # Niveau de complexité du modèle (0, 1 ou 2), 1 = bon compromis perf/précision
    enable_segmentation=False,  # Pas besoin de segmentation du corps ici
    min_detection_confidence=0.5  # Seuil minimum de confiance pour détecter une pose
) as pose:

    """ Boucle tant que la webcam est ouverte et fonctionne """
    
    while cap.isOpened():
        """ Capture une frame (image) depuis la webcam """
        
        ret, frame = cap.read()

        """ Si la lecture a échoué (par ex webcam déconnectée), on arrête le programme """
        
        if not ret:
            print("Erreur: Impossible de lire la vidéo.")
            break

        """ Convertit l’image de BGR (OpenCV) vers RGB (format attendu par MediaPipe) """
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        """ Envoie l’image à MediaPipe pour qu’il détecte les points du corps humain """
        
        results = pose.process(frame_rgb)

        """ Si une pose a été détectée (landmarks disponibles) """
        
        if results.pose_landmarks:
            
            
            """ Affiche les coordonnées de chaque point du squelette détecté """
            
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                print(f"Point {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}, visibility={landmark.visibility:.2f}")

            """ Dessine les points de repère (landmarks) sur l’image originale (en BGR) """
            
            mp_drawing.draw_landmarks(
                frame,  # image où dessiner
                results.pose_landmarks,  # données de position des articulations
                mp_pose.POSE_CONNECTIONS,  # les connexions entre les points (ex : coude-épaule)
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # style des points
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # style des lignes entre points
            )
            

        """ Affiche la frame (avec ou sans squelette dessiné) dans une fenêtre nommée "Pose Detection" """
        
        cv2.imshow("Pose Detection", frame)
        

        """ Attend 10 ms pour permettre l’interaction clavier. Si l’utilisateur appuie sur ‘q’, on quitte la boucle """
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

""" Libère la webcam pour d'autres applications une fois le script terminé """

cap.release()

""" Ferme toutes les fenêtres ouvertes par OpenCV """

cv2.destroyAllWindows()
