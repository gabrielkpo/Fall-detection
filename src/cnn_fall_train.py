#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 21:34:09 2025

@author: gabriel
"""

import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from cnn_fall import FallCNN1D


# === 1. Chargement des données ===

"""
On charge les fichiers numpy contenant les données brutes sous forme de matrices.
- X contient les séquences temporelles de données (30 frames × 132 caractéristiques par frame).
- y contient les labels binaires (0 pour ADL, 1 pour chute).
"""
X = np.load("X.npy")  
y = np.load("y.npy")

"""
Les données numpy sont par défaut en float64 (double précision) qui est coûteux en mémoire et inutile ici.
On convertit en float32 pour réduire la taille mémoire et pour compatibilité avec PyTorch qui préfère ce format.
Cela accélère aussi les calculs.
"""
X = X.astype(np.float32)
y = y.astype(np.float32)  # La BCE Loss de PyTorch attend des labels float (probabilités entre 0 et 1)

# === 2. Split du dataset ===

"""
On divise le dataset en 3 parties :
- 80% pour l'entraînement (train),
- 10% pour la validation (val),
- 10% pour le test (test).
 
Cela permet d'entraîner le modèle, de vérifier son comportement pendant l'entraînement (val),
et enfin de tester sa généralisation finale (test).
"""

# Séparation initiale train (80%) et temporaire (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
"""
stratify=y garantit que la proportion de classes (chute / pas chute) est conservée dans les splits,
évite un déséquilibre non voulu dans les sous-ensembles.
random_state fixe la graine aléatoire pour reproductibilité.
"""

# Division de la partie temporaire en val (10%) et test (10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
# test_size=0.5 pour couper la partie 20% en deux parts égales


# === 3. Dataloaders PyTorch ===

"""
On convertit les arrays numpy en tenseurs PyTorch, puis on crée des Dataset et DataLoader.
Les DataLoader permettent de générer des batchs pour l'entraînement.
- batch_size=32 est un compromis classique entre vitesse et usage mémoire.
- shuffle=True sur l'entraînement pour casser l'ordre des données, éviter un apprentissage biaisé.
"""

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# === Instanciation du modèle, critère de perte et optimiseur ===

"""
On crée une instance du réseau CNN 1D :
- input_channels=132 correspond aux caractéristiques par frame (ex : 33 points × 4 coordonnées)
- seq_length=30 correspond au nombre de frames dans chaque séquence
"""
model = FallCNN1D(input_channels=132, seq_length=30)

"""
Critère de perte : Binary Cross Entropy Loss
- adapté aux problèmes de classification binaire.
- compare la sortie (probabilité entre 0 et 1) avec le label réel (0 ou 1).
"""
criterion = nn.BCELoss()

"""
Optimiseur Adam : méthode stochastique basée sur le gradient adaptatif.
- lr=0.001 est un taux d'apprentissage standard.
- Adam combine avantages d'Adamax, RMSProp, accélère convergence sans trop de réglages.
"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# === 4. Boucle d'entraînement principale ===

num_epochs = 150  # Nombre d'itérations complètes sur tout le dataset d'entraînement.
"""
Plus d'époques peuvent améliorer la performance mais aussi risquer du sur-apprentissage (overfitting).
On surveillera la perte de validation pour détecter cela.
"""

# Paramètres pour early stopping
best_val_loss = float('inf')
patience = 50
counter = 0

for epoch in range(num_epochs):
    model.train()  # Passage en mode entraînement (activate dropout, batchnorm, etc.)
    train_loss = 0.0  # Accumulateur pour la perte moyenne sur epoch
    correct = 0  # Compteur pour le nombre de prédictions correctes
    total = 0    # Nombre total d'exemples vus

    for inputs, labels in train_loader:
        """
        inputs shape : (batch_size, seq_length, input_channels) → on doit permuter pour Conv1d PyTorch
        Conv1d attend (batch_size, input_channels, seq_length).
        La permutation est obligatoire sinon la convolution est sur la mauvaise dimension.
        """
        inputs = inputs.permute(0, 2, 1)  # Maintenant shape = (batch_size, 132, 30)

        optimizer.zero_grad()  # Remise à zéro des gradients du batch précédent

        outputs = model(inputs).squeeze()
        """
        outputs shape : (batch_size, 1) → squeeze pour obtenir (batch_size,)
        On doit avoir le même shape que labels pour la perte.
        """

        loss = criterion(outputs, labels)  # Calcul de la perte entre sortie et vérité terrain

        loss.backward()  # Calcul des gradients par rétropropagation
        optimizer.step()  # Mise à jour des poids avec Adam et gradients calculés

        # On cumule la perte pondérée par la taille du batch (pour moyenne correcte)
        train_loss += loss.item() * inputs.size(0)

        # Seuil à 0.5 pour convertir la probabilité en classe binaire
        predicted = (outputs > 0.5).float()

        # Comptage des prédictions correctes
        correct += (predicted == labels).sum().item()
        total += labels.size(0)  # Nombre d'exemples dans ce batch

    avg_train_loss = train_loss / total  # Perte moyenne par exemple sur l'epoch
    train_acc = correct / total          # Précision moyenne sur l'epoch

    # === Validation (pas de mise à jour du modèle) ===
    model.eval()  # Mode évaluation désactive dropout, batchnorm fixées, pas de calcul de gradient

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():  # Désactive le calcul des gradients pour économiser la mémoire et le calcul
        for inputs, labels in val_loader:
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"[{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # === Early stopping ===
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")  # Sauvegarde du meilleur modèle
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
