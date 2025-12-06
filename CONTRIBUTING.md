# CONTRIBUTING

Ce document décrit les règles de base pour contribuer à ce projet, que ce soit manuellement ou via un agent (Codex / ChatGPT GitHub connector).

## Branches

- La branche par défaut est `main`.
- Toute nouvelle fonctionnalité ou refactor doit se faire sur une branche :
  - `feature/<nom-court>` pour les fonctionnalités,
  - `fix/<nom-court>` pour les corrections de bugs.

Exemples :

- `feature/pyannote-diarization-tuning`
- `feature/xtts-speaker-cache`
- `fix/nllb-batch-size`

## Processus de modification

1. Créer une branche à partir de `main`.
2. Effectuer les modifications (scripts, config, docs).
3. Vérifier que :
   - les scripts s’exécutent au moins sur un épisode test,
   - aucune donnée volumineuse n’a été ajoutée au dépôt.
4. Mettre à jour la documentation si nécessaire (`README.md`, `AGENTS.md`).
5. Créer une Pull Request vers `main` avec :
   - un résumé clair,
   - une liste des scripts / modules modifiés,
   - les éventuels changements de format de données.

## Style de code

- Python 3.10
- Commentaires et docstrings en français si possible.
- Garder les fonctions relativement courtes et lisibles.
- Préférer la clarté à l’optimisation prématurée.

## Données

- Ne jamais versionner les fichiers vidéo ou audio :
  - `data/episodes_raw/`
  - `data/audio_raw/`
  - `data/audio_stems/`
  - `data/dub_audio/`
  - `data/episodes_dubbed/`

Ces répertoires sont ignorés via `.gitignore` et doivent le rester.

---

Merci de respecter ces règles pour garder le projet propre et maintenable.

