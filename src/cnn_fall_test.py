#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jun  1 21:34:09 2025

@author: gabriel

Pour exécuter le script dans le terminal : 
 
cd ~/Desktop/Perso/projet_info/Reco_mouv/cnn_fall
python3 cnn_fall_test.py

"""


import torch
import numpy as np
from cnn_fall import FallCNN1D
import cv2
import mediapipe as mp
from collections import deque

# === Charger le modèle ===
model = FallCNN1D(input_channels=132, seq_length=30)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# === Prédiction ===
def predict_fall(sequence_30x132):
    x = torch.tensor(sequence_30x132, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
    with torch.no_grad():
        output = model(x).item()
    return 1 if output > 0.5 else 0, output  # 1 = chute, 0 = ADL

# === Initialisation ===
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
                else:
                    cv2.putText(frame, f"ADL ({prob:.2f})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
