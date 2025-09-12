# Organisation du projet

## Dossiers principaux

- `src/` : Modules Python réutilisables (modèles, utilitaires, etc.)
- `scripts/` : Scripts d'exécution (entraînement, test, prédiction)
- `data/raw/` : Données brutes (archives, dossiers originaux)
- `data/processed/` : Données traitées (fichiers .npy, etc.)
- `data/events/` : Événements de chute horodatés
- `data/videos/` : Vidéos de test
- `models/` : Modèles entraînés
- `notebooks/` : Notebooks Jupyter

## Utilisation

- Placez tout nouveau script dans `scripts/`.
- Placez les modules réutilisables dans `src/`.
- Les données brutes doivent aller dans `data/raw/`, les données traitées dans `data/processed/`.
- Les modèles sauvegardés dans `models/`.
- Les vidéos de test dans `data/videos/`.

## Conseils

- Ajoutez vos dépendances dans un fichier `requirements.txt`.
- Utilisez `.gitignore` pour ignorer les fichiers volumineux ou temporaires.

---
Organisation proposée par GitHub Copilot.