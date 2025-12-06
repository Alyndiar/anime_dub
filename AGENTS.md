# AGENTS.md – Instructions pour Codex / ChatGPT GitHub Connector

Ce dépôt contient un pipeline de **doublage local d’animés chinois en français** (ZH → FR) utilisant Whisper, NLLB, pyannote.audio, XTTS-v2 et ffmpeg.

Ces instructions s’adressent à l’agent d’aide au développement (Codex / ChatGPT) connecté à ce dépôt.

---

## Objectifs du projet

- Transformer des épisodes d’animé chinois (vidéos locales) en **versions doublées en français**, en conservant :
  - la **synchronisation temporelle** des dialogues,
  - un **casting de voix cohérent** (une voix par personnage, clonée à partir de la VO),
  - autant que possible le **mixage BGM+SFX original**.

- Fournir un pipeline **scriptable et automatisable** pour traiter des centaines d’épisodes.

---

## Contexte technique

- OS principal : Windows 11.
- GPU : NVIDIA RTX 4080 (CUDA).
- Langage principal : **Python 3.10**.
- Outils principaux :
  - `ffmpeg` pour audio/vidéo,
  - `faster-whisper` pour transcription chinois,
  - `NLLB-200` (transformers) pour traduction zh→fr,
  - `pyannote.audio` pour diarisation et embeddings de locuteur,
  - `TTS` (Coqui, modèle XTTS-v2) pour TTS multi-voix en français,
  - scripts Python dans `scripts/`.

Le dossier `data/` contient **uniquement des chemins et des artefacts locaux** non versionnés (vidéos, audios, sorties intermédiaires).

---

## Règles de contribution pour l’agent

### 1. Branches & PR

- **Ne pas travailler directement sur `main`.**
- Toujours créer / utiliser une branche `feature/...` :
  - Exemple : `feature/whisper-config`, `feature/diarization-improvements`, `feature/xtts-batching`.
- Pour toute modification conséquente (refactor, nouvelle étape pipeline), proposer un **plan** dans la description de PR :
  - Résumé de ce qui est changé,
  - Scripts impactés,
  - Effets sur les chemins de données / configs.

### 2. Fichiers à éviter / à ne jamais créer

- Ne jamais ajouter de fichiers binaires volumineux dans le repo :
  - pas de vidéos (`.mkv`, `.mp4`),
  - pas d’audio (`.wav`, `.flac`, etc.),
  - pas de sorties générées (`data/audio_raw/`, `data/dub_audio/`, etc.).

Ces répertoires sont **ignorés** via `.gitignore` et doivent le rester.

### 3. Style de code

- Langue des commentaires / docstrings : **français** de préférence, clair et concis.
- Respecter la structure existante :
  - chaque script `scripts/0X_*.py` doit pouvoir être lancé **indépendamment**,
  - ne pas casser les chemins `data/...` et `config/...` sans mettre à jour le README.
- Limiter les dépendances :
  - privilégier les libs déjà utilisées (`numpy`, `pydub`, `soundfile`, `transformers`, etc.),
  - éviter d’ajouter une nouvelle grosse dépendance sans réelle nécessité.

### 4. Approche de développement

- Toujours :
  1. Faire un **plan en plusieurs étapes** avant de modifier des fichiers (dans la conversation / PR).
  2. Procéder par **petites étapes** (commits plus fréquents plutôt qu’un énorme commit), mais ne pas toujours se limiter à la **modification minimale**. Evaluer au cas par cas les changements nécessaires pour fournir une **implémentation complète** des fonctionnalités demandées, incluant un **refactor** ou des **changements aux APIs et formats de données** lorsque jugé préférable
  3. Quand un refactor ou un changement de format de données est nécessaire :
     - s’assurer de propager le changement à **tous les scripts concernés**,
     - mettre à jour la documentation correspondante (`README`, commentaires, docstrings).

- Préserver la lisibilité :
  - ajouter / mettre à jour les commentaires sur les sections critiques,
  - expliquer brièvement les choix de paramètres (seuils de similarité, tailles de lots, etc.).

---

## Priorités pour l’assistant

Quand l’assistant propose des modifications ou du code, il doit :

1. **Respecter le pipeline existant**  
   Ne pas supprimer une étape sans raison (ex. ne pas supprimer la diarisation juste pour “simplifier”).

2. **Privilégier la robustesse** sur la performance micro-optimisée :
   - Le but est un pipeline *fiable* qu’on peut faire tourner sur de longues séries, pas uniquement un bench de vitesse.

3. **Préserver la modularité** :
   - Chaque script doit rester exécutable depuis la ligne de commande,
   - Les fonctions réutilisables peuvent être factorisées dans un module utilitaire (ex. `scripts/utils.py`) si besoin.

4. **Documenter tout changement de format** :
   - Si la structure de `data/segments/*_segments.json` ou `config/characters.yaml` change, mettre à jour `README.md` et ajouter quelques exemples.

---

## Points spécifiques pour ce projet

### Diarisation & banque de personnages

- Utiliser pyannote.audio pour diarisation et extraction d’embeddings vocaux.
- Construire la banque de personnages comme une **collection de profils** :
  - un personnage peut avoir plusieurs profils (jeune, adulte, changement d’acteur),
  - chaque profil est représenté par un embedding moyen.
- L’assistant doit :
  - éviter de mélanger tous les personnages dans un seul embedding,
  - proposer des structures claires (YAML + `.npy`).

### Synthèse XTTS-v2

- Toujours rappeler la **limitation non commerciale** de XTTS-v2.
- Utiliser la fonctionnalité **`speaker_wav`** pour cloner les voix à partir de clips extraits de la VO.
- Gérer le **taux d’échantillonnage** correctement (XTTS-v2 est en 24 kHz), et adapter ceci dans le mix final.

### Respect du cadre légal

- Rappel : ce projet est prévu pour un **usage privé / accessibilité**.
- Ne jamais ajouter au repo de données qui pourraient poser problème (vidéos, pistes audio complètes, etc.).

---

## Ce que l’assistant peut faire

- Générer / modifier des scripts Python dans `scripts/`.
- Proposer des améliorations de structure (ex. factorisation dans un module utilitaire).
- Ajouter ou améliorer la documentation (`README`, `AGENTS`, `CONTRIBUTING`).
- Aider à paramétrer les modèles (Whisper, NLLB, XTTS, pyannote).

## Ce que l’assistant ne doit pas faire

- Ajouter des ressources audio/vidéo au dépôt.
- Supprimer des protections de licence ou ignorer les restrictions d’usage des modèles tiers.
- Écrire du code qui tente de contourner des DRM ou de récupérer du contenu qu’on n’a pas le droit de traiter.

---

En résumé :  
L’agent doit aider à rendre ce pipeline **plus robuste, plus clair et plus facile à maintenir**, en respectant la structure actuelle, les contraintes matérielles et légales, et en documentant clairement tout changement.
