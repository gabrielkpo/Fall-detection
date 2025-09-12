#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 21:33:17 2025

@author: gabriel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FallCNN1D(nn.Module):
    
    def __init__(self, input_channels, seq_length):
        """
        Classe de détection de chute basée sur un CNN 1D.

        input_channels : nombre de "capteurs" d'entrée (ex : 33 points MediaPipe × 4 coordonnées = 132).
        seq_length : longueur de la séquence temporelle (nombre de frames analysées, ex : 30 frames à 30 FPS).
        """
        super(FallCNN1D, self).__init__()

        """
        Couche convolutionnelle 1 : détection des patterns rapides
        - in_channels = input_channels (132)
        - out_channels = 32 → cela signifie qu'on apprend 32 motifs spatiaux différents
        - kernel_size = 5 → la convolution regarde 5 frames adjacentes (utile pour capter les transitions de mouvement)
        - padding = 2 permet de conserver la longueur temporelle en sortie égale à celle en entrée
        """
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2)

        """
        Couche convolutionnelle 2 : combinaisons des patterns pour former des evènements
        - reprend les 32 canaux de sortie de la couche 1
        - continue d’apprendre des motifs plus complexes (hiérarchiques)
        """
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        """
        Couche convolutionnelle 3 : fusion des evènements pour detecter la chute complète
        - encore 32 canaux, kernel de taille 5
        - cette 3e couche permet d’augmenter la "vision temporelle" du réseau sur la séquence
        - si on mettait moins de couches : moins de hiérarchie, donc moins de sensibilité aux chutes progressives
        - si on mettait plus de couches : plus puissant mais plus lent, risque de surapprentissage si peu de données
        """
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        """
        Global pooling :
        - AdaptativeMaxPool1d(output_size=1) compresse chaque séquence temporelle en un seul score par canal
        - Cela résume l’information temporelle tout en gardant les 32 canaux
        - On pourrait utiliser un moyennage au lieu du max, mais le max est plus sensible aux événements intenses
        """
        self.global_pool = nn.AdaptiveMaxPool1d(output_size=1)

        """
        Couche fully connected :
        - input = 32 (nombre de canaux en sortie du pooling)
        - output = 1 (probabilité de chute entre 0 et 1)
        - Cette couche apprend à combiner les motifs détectés par les 3 couches CNN
        """
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        """
        Définition du passage avant :
        x est un tenseur de forme (batch_size, input_channels, seq_length)
        """

        """
        Chaque couche convolutive est suivie d'une fonction d'activation ReLU
        ReLU = Rectified Linear Unit : met toutes les valeurs négatives à 0
        Cela introduit de la non-linéarité et aide à apprendre des représentations plus complexes
        """
        x = F.relu(self.conv1(x))  # sortie (batch_size, 32, seq_length)
        x = F.relu(self.conv2(x))  # idem
        x = F.relu(self.conv3(x))  # idem

        """
        Global pooling : réduit la dimension temporelle (seq_length → 1)
        Résultat : (batch_size, 32, 1)
        """
        x = self.global_pool(x)

        """
        Suppression de la dernière dimension (inutile) pour passer à la couche fully connected
        Résultat : (batch_size, 32)
        """
        x = x.squeeze(-1)

        """
        Dernière couche linéaire + activation sigmoid :
        - Renvoie une probabilité entre 0 et 1
        - On pourra ensuite appliquer un seuil (ex: > 0.5 = chute détectée)
        """
        x = self.fc(x)
        return torch.sigmoid(x)
