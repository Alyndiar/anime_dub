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
│  ├─ 02_diarize.py
│  ├─ 03_whisper_transcribe.py
│  ├─ 04_translate_nllb.py
│  ├─ 05_build_speaker_bank.py
│  ├─ 06_assign_characters.py
│  ├─ 07_synthesize_xtts.py
│  ├─ 08_mix_audio.py
│  ├─ 09_remux.py
│  └─ gui_pipeline.py        # GUI pour orchestrer les étapes 01→09
└─ config/
   ├─ paths.yaml
   ├─ characters.yaml
   └─ xtts_config.yaml

Les scripts utilisent `scripts/utils_config.py` pour charger ces fichiers de config et résoudre les chemins depuis la racine du projet. Les fonctions principales :

- `get_data_path(key)`: récupère un `Path` à partir d'une clef définie dans `config/paths.yaml`.
- `ensure_directories([...])`: crée (si besoin) les répertoires référencés et renvoie leur mapping.
- `load_characters_config()` / `load_xtts_config()`: chargent les paramètres vocaux et XTTS.

Une interface GUI est disponible via `python scripts/gui_pipeline.py` :

- menus pour lancer les étapes 01→09 avec arrêt automatique après chaque fichier, répertoire ou étape ;
- modification en direct des chemins définis dans `config/paths.yaml` (base par défaut = racine du projet) ;
- sauvegarde et rechargement de l'état (chemins, thème sombre/clair, étape et fichier en cours) dans `config/gui_state.json` ou via « Fichier → Enregistrer sous… » ;
- reprise exacte d'une exécution interrompue grâce aux options `--stem` ajoutées sur les scripts (par exemple `python scripts/03_whisper_transcribe.py --stem episode_001`).
