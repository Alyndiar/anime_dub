# Local Anime Dubbing Pipeline (ZH → FR, multi-voix)

Ce projet est un pipeline local pour **doubler des animés chinois en français**, en utilisant :

- **faster-whisper** pour la transcription mandarin,
- **NLLB-200** (Meta) pour la traduction chinois → français,
- **pyannote.audio** pour la diarisation (qui parle quand),
- une **banque de personnages** basée sur des empreintes vocales,
- **XTTS-v2 (Coqui TTS)** pour générer des voix françaises clonées à partir des voix originales,
- **ffmpeg** pour le mixage audio et le remux vidéo.

> ⚠️ Usage strictement **privé / non commercial**.  
> XTTS-v2 est sous licence Coqui Public Model License 1.0 (non commercial).  
> À ne pas utiliser pour des distributions publiques ou commerciales.

---

## Objectifs

- Permettre la création d’un **doublage français local** pour de longues séries d’animés chinois (200–300 épisodes).
- **Aider une personne malvoyante** à suivre les dialogues grâce à un audio français synchronisé.
- Minimiser les coûts de services en ligne en utilisant des modèles open-source locaux.

---

## Architecture générale

Pipeline pour chaque épisode :

1. **Extraction audio**
   - `ffmpeg` extrait l’audio de la vidéo source (`data/episodes_raw/` → `data/audio_raw/`).
   - Formats vidéo détectés : `.mkv`, `.mp4`, `.mov`, `.m4v`, `.avi`, `.ts`.

2. **(Optionnel) Séparation stems**  
   - Utilisation d’outils comme **UVR / Demucs** pour séparer voix / BGM+SFX.

3. **Diarisation**  
   - `pyannote.audio` produit des segments temporels par locuteur (`data/diarization/`).

4. **Transcription mandarin**  
   - `faster-whisper` génère des segments texte + timestamps (`data/transcripts/whisper_json/`, `.srt` zh).

5. **Traduction vers le français**  
   - `NLLB-200` (via `transformers`) traduit les segments chinois → français (`data/transcripts/fr_srt/`).

6. **Banque de personnages & reconnaissance de locuteur**  
   - Embeddings de voix (pyannote/embedding) → **clusters / profils de personnage** dans `data/speaker_bank/`.
   - Attribution automatique des segments à chaque personnage (`data/segments/`).

7. **Synthèse vocale FR**  
   - **XTTS-v2** génère les dialogues en français, avec une **voix par personnage** (`data/dub_audio/*_fr_voices.wav`).

8. **Mixage audio**  
   - Mix BGM+SFX original + voix FR (`data/dub_audio/*_fr_full.wav`).

9. **Remux vidéo**  
   - `ffmpeg` remux la vidéo d’origine avec la **nouvelle piste audio FR** (`data/episodes_dubbed/`).

---

## Arborescence

Arborescence principale :

```text
anime-dub/
├─ launcher.bat             # Lance le GUI directement (Windows)
├─ data/
│  ├─ episodes_raw/          # Vidéos source (non versionnées)
│  ├─ audio_raw/             # Audios extraits
│  ├─ audio_stems/           # Stems (vocals, instrumental)
│  ├─ diarization/           # Résultats pyannote
│  ├─ transcripts/
│  │  ├─ whisper_json/       # Segments Whisper (ZH)
│  │  ├─ whisper_json_fr/    # Segments Whisper traduits (ZH→FR)
│  │  ├─ zh_srt/             # Sous-titres ZH
│  │  └─ fr_srt/             # Sous-titres FR
│  ├─ segments/              # Segments texte + personnage
│  ├─ voices/                # Clips de référence voix par personnage
│  ├─ speaker_bank/          # Embeddings voix par personnage/profil
│  ├─ dub_audio/             # Pistes audio FR générées
│  └─ episodes_dubbed/       # Vidéos remuxées avec piste FR
├─ scripts/
│  ├─ 01_extract_audio.py
│  ├─ 02_separate_stems.py
│  ├─ 03_diarize.py
│  ├─ 04_whisper_transcribe.py
│  ├─ 05_translate_nllb.py
│  ├─ 06_build_speaker_bank.py
│  ├─ 07_assign_characters.py
│  ├─ 08_synthesize_xtts.py
│  ├─ 09_mix_audio.py
│  ├─ 10_remux.py
│  └─ gui_pipeline.py        # GUI pour orchestrer les étapes 01→10
└─ config/
   ├─ paths.yaml
   ├─ characters.yaml
   └─ xtts_config.yaml

Les scripts utilisent `scripts/utils_config.py` pour charger ces fichiers de config et résoudre les chemins depuis la racine du projet. Les fonctions principales :

- `get_data_path(key)`: récupère un `Path` à partir d'une clef définie dans `config/paths.yaml`.
- `ensure_directories([...])`: crée (si besoin) les répertoires référencés et renvoie leur mapping.
- `load_characters_config()` / `load_xtts_config()`: chargent les paramètres vocaux et XTTS.

Une interface GUI est disponible via `python scripts/gui_pipeline.py` ou directement avec `launcher.bat` sous Windows :

- menus pour lancer les étapes 01→10 avec arrêt automatique après chaque fichier, répertoire ou étape ;
- ciblage des épisodes : traitement complet, mode « 1 seul épisode » avec sélection de fichier dédiée ou sélection multi-fichiers via un explorateur ;
- gestion de projets (Créer/Charger/Sauvegarder/Fermer) avec un répertoire de base par projet, ses fichiers de config dédiés dans `<projet>/config/` et un état persistant par projet ;
- création guidée : saisie du titre d'animé et du nom de projet, proposition d'un dossier dédié (inexistant) pré-rempli avec le fichier `<nom_du_projet>.yaml`, les configs par défaut et la hiérarchie `data/` ;
- modification hiérarchique des chemins via le menu « Options → Configurer les chemins… » : chaque entrée dispose d'un bouton « … » qui ouvre un sélecteur ancré sur le répertoire parent résolu ;
- sauvegarde et rechargement de l'état (chemins, thème sombre/clair, étape et fichier en cours) dans `config/gui_state.json` ou via « Fichier → Enregistrer sous… » ;
- lors de l'exécution d'une étape, le GUI exporte les variables d'environnement `ANIME_DUB_PROJECT_ROOT` et `ANIME_DUB_CONFIG_DIR` pour que les helpers (`get_data_path`, `ensure_directories`, etc.) résolvent correctement les chemins du projet sélectionné ;
- option « Verbose » dans le menu Options ou la barre de commandes pour tracer en détail les appels des scripts (commande, environnement `ANIME_DUB_VERBOSE`, paramètre `--verbose` lorsque disponible) ;
- les logs sont affichés dans la fenêtre et simultanément écrits dans `<base_du_projet>/logs/gui_pipeline.log` pour conserver une trace des commandes et sorties ;
- reprise exacte d'une exécution interrompue grâce aux options `--stem` ajoutées sur les scripts (par exemple `python scripts/04_whisper_transcribe.py --stem episode_001`).

### Étape 02 : séparation voix / instrumental

- Entrée attendue : les WAV complets (`*_full.wav`) produits par `01_extract_audio` dans `data/audio_raw/`.
- Sortie : deux fichiers par stem dans `data/audio_stems/` (`<stem>_vocals.wav` et `<stem>_instrumental.wav`).
- Outil recommandé : **Demucs** (two-stems vocals) pour limiter les dépendances :

```bash
pip install --upgrade demucs  # installe également PyTorch/torchaudio
python scripts/02_separate_stems.py --tool demucs --demucs-model htdemucs --demucs-device cuda
```

- Alternative UVR : si vous disposez d'une commande UVR (CLI portable ou pip) acceptant les arguments d'entrée/sortie, passez-la via `--tool uvr` et le template `--uvr-command` (placeholders `{input}`, `{output_dir}`, `{model}`), par exemple :

```bash
python scripts/02_separate_stems.py \
  --tool uvr \
  --uvr-command "python inference.py --input {input} --output_dir {output_dir} --model {model}" \
  --uvr-model C:/models/uvr5/model.pth \
  --uvr-vocals-name vocals.wav \
  --uvr-instrumental-name instrumental.wav
```

Dans les deux cas, la commande accepte `--stem` pour cibler un épisode et `--overwrite` pour régénérer des stems déjà présents.

### CLI ou GUI ? Pourquoi conserver les deux

- Les scripts `scripts/0X_*.py` restent **exécutables en ligne de commande** (contrainte historique du projet) : chaque script gère ses propres arguments (`--stem`, `--verbose`, etc.) et fonctionne sans le GUI. Cela reste indispensable pour les usages batch, le débogage ciblé et l’exécution sur des machines sans environnement graphique.
- Le GUI **orchestration** sert de surcouche ergonomique : il prépare l’environnement (variables `ANIME_DUB_PROJECT_ROOT` / `ANIME_DUB_CONFIG_DIR`), sélectionne les stems ciblés et enchaîne les scripts en respectant les pauses, le mode verbose et la reprise. Il ne remplace pas la logique métier des scripts mais la pilote.
- Choix de design :
  - **Facteurs communs** dans `scripts/utils_config.py` ou des fonctions internes : cela évite le double code et permet au GUI d’importer et d’appeler proprement sans casser l’interface CLI.
  - **Journalisation unifiée** : chaque script utilise `logging` (et accepte `--verbose` / `ANIME_DUB_VERBOSE`) pour que le GUI ou la CLI capturent la même trace.
  - **Isolation par projet** : les chemins par projet sont résolus via les variables d’environnement exportées par le GUI, mais un opérateur CLI peut toujours définir ces variables ou utiliser les arguments `--stem` pour cibler un épisode précis.
- En pratique :
  - continuer à exposer un `if __name__ == "__main__":` pour conserver l’entrée CLI ;
  - structurer les scripts avec des fonctions réutilisables (ex. `run_extract_audio(stems, config, logger)`) afin que le GUI puisse les importer directement si besoin, sans empêcher l’appel direct en CLI ;
  - limiter les effets de bord (globales) pour que le passage de contexte GUI/CLI reste prévisible.
