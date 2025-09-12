#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:09:10 2025

@author: gabriel

Pour exécuter le script dans le terminal : 
cd ~/Desktop/Perso/projet_info/Reco_mouv/cnn_fall
python3 cnn_fall_test_song_notif.py
"""

import os
import datetime
import platform
import torch
import numpy as np
from cnn_fall import FallCNN1D
import cv2
import mediapipe as mp
from collections import deque
import pygame
import time

# Init pygame pour son
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

def start_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # boucle infinie

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Fonction notification macOS native
if platform.system() == "Darwin":
    from Cocoa import NSUserNotification, NSUserNotificationCenter

    def notify_mac(title, subtitle, message):
        notification = NSUserNotification.alloc().init()
        notification.setTitle_(title)
        notification.setSubtitle_(subtitle)
        notification.setInformativeText_(message)
        notification.setSoundName_("NSUserNotificationDefaultSoundName")
        NSUserNotificationCenter.defaultUserNotificationCenter().deliverNotification_(notification)
else:
    def notify_mac(title, subtitle, message):
        pass

# Charger modèle
model = FallCNN1D(input_channels=132, seq_length=30)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

def predict_fall(sequence_30x132):
    x = torch.tensor(sequence_30x132, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
    with torch.no_grad():
        output = model(x).item()
    return 1 if output > 0.5 else 0, output  # 1 = chute, 0 = ADL

# Variables globales pour gérer chute et captures
fall_alarm_on = False
capture_folder = None
last_capture_time = 0

def handle_fall_alert(frame):
    global capture_folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    capture_folder = f"fall_event_{timestamp}"
    os.makedirs(capture_folder, exist_ok=True)
    print(f"[INFO] Nouveau dossier de capture créé : {capture_folder}")

    # Sauvegarde image initiale
    filename = os.path.join(capture_folder, f"fall_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[INFO] Image sauvegardée sous : {filename}")

    # Notification système
    try:
        if platform.system() == "Darwin":
            notify_mac("Alerte", "Chute détectée !", "Attention, une chute a été détectée.")
        elif platform.system() == "Linux":
            os.system(f'notify-send "Alerte" "Chute détectée !"')
        elif platform.system() == "Windows":
            from win10toast import ToastNotifier
            ToastNotifier().show_toast("Alerte", "Chute détectée !", duration=5)
    except Exception as e:
        print(f"[Erreur notification] : {e}")

# Initialisation caméra et MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
sequence_buffer = deque(maxlen=30)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            features = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z, lm.visibility)]
            sequence_buffer.append(features)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            if len(sequence_buffer) == 30:
                

                sequence_np = np.array(sequence_buffer)
                label, prob = predict_fall(sequence_np)

                if prob > 0.8:

                    cv2.putText(frame, "CHUTE", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                    if not fall_alarm_on:
                        print("Chute détectée, alarme activée")
                        start_alarm()
                        handle_fall_alert(frame)
                        fall_alarm_on = True
                        last_capture_time = time.time()

                    else:
                        # Toutes les 1s, sauvegarder une capture supplémentaire
                        current_time = time.time()
                        if current_time - last_capture_time >= 1.0 and capture_folder is not None:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            filename = os.path.join(capture_folder, f"fall_{timestamp}.jpg")
                            cv2.imwrite(filename, frame)
                            print(f"[INFO] Capture supplémentaire sauvegardée sous : {filename}")
                            last_capture_time = current_time

                else:
                    cv2.putText(frame, f"ADL ({prob:.2f})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                    if fall_alarm_on:
                        print("Position normale détectée, arrêt de l'alarme")
                        stop_alarm()
                        fall_alarm_on = False
                        capture_folder = None

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
