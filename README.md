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
- les logs s'affichent en continu pendant l'exécution (stdout/stderr streaming) et sont simultanément écrits dans `<base_du_projet>/logs/gui_pipeline.log` pour conserver une trace des commandes et sorties ;
- reprise exacte d'une exécution interrompue grâce aux options `--stem` ajoutées sur les scripts (par exemple `python scripts/04_whisper_transcribe.py --stem episode_001`).

### Étape 02 : séparation voix / instrumental

- Entrée attendue : les WAV complets (`*_full.wav`) produits par `01_extract_audio` dans `data/audio_raw/`.
- Sortie : deux fichiers par stem dans `data/audio_stems/` (`<stem>_vocals.wav` et `<stem>_instrumental.wav`).
- Outil recommandé : **Demucs** (two-stems vocals) pour limiter les dépendances :

```bash
pip install --upgrade demucs  # installe également PyTorch/torchaudio
python scripts/02_separate_stems.py --tool demucs --demucs-model htdemucs
```

- Pour exploiter le GPU (ex. RTX 4080), installez une build CUDA de PyTorch (sinon Demucs repassera en CPU et journalisera `torch.version.cuda=None`). Exemple pour CUDA 12.1 :

```bash
pip install --upgrade "torch==2.5.1+cu121" "torchaudio==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
```

Vérifiez ensuite :

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
PY
```

- Si vous utilisez déjà une build GPU (ex. `torch==2.8.0+cu128`) mais que Demucs bascule en CPU, forcez la vérification via :

```bash
python scripts/02_separate_stems.py --demucs-device cuda --demucs-require-cuda --stem mon_episode --verbose
```

Cette commande lèvera une erreur explicite si CUDA est indisponible dans l'environnement du script. Vérifiez alors que le driver NVIDIA est installé et que `demucs` est bien exécuté dans la même venv/conda que votre PyTorch CUDA.

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
`--demucs-device` vaut `auto` par défaut : si CUDA est indisponible (PyTorch compilé CPU), le script bascule automatiquement sur
le CPU pour éviter un échec de séparation ; utilisez `--demucs-require-cuda` pour forcer un échec explicite lorsque le GPU n'est
pas utilisable.

### Résoudre les conflits pip (numpy/pandas avec gruut et TTS)

Si `pip check` ou l’installation affiche des messages du type :

```
gruut 2.2.3 has requirement numpy<2.0.0,>=1.19.0, but you have numpy 2.2.6.
tts 0.22.0 has requirement numpy==1.22.0; python_version <= "3.10", but you have numpy 2.2.6.
tts 0.22.0 has requirement pandas<2.0,>=1.4, but you have pandas 2.3.3.
```

alignez les versions pour respecter les contraintes de gruut/TTS (et éviter de casser les autres dépendances) :

1. Dans l’environnement actuel, repassez sur des versions compatibles puis vérifiez avec `pip check` :

```bash
pip install --upgrade "numpy==1.22.0" "pandas>=1.4,<2.0" "tts==0.22.0" "gruut==2.2.3"
pip check
```

2. Si d’autres paquets nécessitent numpy ≥2.x, créez un environnement dédié pour le pipeline (Python 3.10 recommandé), installez
   d’abord les dépendances de base (ffmpeg, demucs, etc.), puis appliquez les versions compatibles ci-dessus. Exemple conda :

```bash
conda create -n anime_dub_py310 python=3.10
conda activate anime_dub_py310
pip install --upgrade "numpy==1.22.0" "pandas>=1.4,<2.0" "tts==0.22.0" "gruut==2.2.3"
pip check
```

Gardez les installations critiques (torch/torchaudio/torchvision, demucs, TTS) dans le même environnement pour éviter les
incohérences, et réexécutez `pip check` après chaque mise à jour majeure.

### Environnement dédié pour la diarisation (PyTorch / pyannote / ffmpeg)

`pyannote.audio` essaie d’utiliser `torchcodec` pour décoder les WAV. Sur Windows, aucune roue torchcodec compatible CUDA 12.1 n’est actuellement publiée sur PyPI, ce qui génère des avertissements au lancement. Le script `03_diarize.py` contourne automatiquement l’absence de torchcodec en préchargeant l’audio via `torchaudio` (fallback `soundfile`). Pour isoler la pile PyTorch/pyannote/ffmpeg, utilisez un environnement conda séparé, pré-configuré dans `config/diarization_env.yml` :

```bash
# Création (une seule fois)
conda env create -f config/diarization_env.yml

# Activation avant de lancer le GUI ou 03_diarize.py
conda activate anime_dub_diar
```

> ⚠️ Si la résolution échoue avec un message `cuda-nvtx >=12.1,<12.2` introuvable, vérifiez que le canal `nvidia` est bien activé (il est déjà listé dans `config/diarization_env.yml`). Vous pouvez l’ajouter globalement au besoin :
>
> ```bash
> conda config --add channels nvidia
> ```

Cet environnement fournit PyTorch 2.2.2 (CUDA 12.1), pyannote.audio 3.1.1 et ffmpeg 6. torchcodec n’y est **pas** installé par défaut pour éviter les échecs Windows ; s’il devient disponible ultérieurement (roue compatible Windows/CUDA), installez-le dans cet environnement sans toucher à PyTorch :

```bash
pip install --upgrade --no-deps torchcodec
```

Pour vérifier la stack audio avant d’exécuter la diarisation :

```bash
python - <<'PY'
import torch
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)
import pyannote.audio
print("pyannote.audio", pyannote.audio.__version__)
try:
    import torchcodec
    print("torchcodec", torchcodec.__version__)
except ImportError:
    print("torchcodec absent : fallback torchaudio/soundfile actif pour le décodage")
PY
```

**Compatibilité pyannote :** le fichier `config/diarization_env.yml` inclut `pyannote.audio==3.1.1` et les dépendances critiques (PyTorch 2.2.2 CUDA 12.1, ffmpeg 6). torchcodec reste optionnel grâce au préchargement audio dans `03_diarize.py`. Si vous installez torchcodec manuellement, vérifiez la cohérence torch/ffmpeg indiquée dans sa documentation. Si vous devez utiliser une autre version de pyannote, mettez à jour `config/diarization_env.yml` en conséquence et gardez une pile PyTorch/ffmpeg cohérente.

Lancez ensuite la diarisation sur un épisode :

```bash
python -u scripts/03_diarize.py --stem "Soul land episode 01 vostfr"
```

**Questions fréquentes :**

- **Qu’est-ce que gruut ?** Bibliothèque de **génération phonémique** (tokenisation, phonétisation) utilisée par Coqui TTS ; la
  dernière version publiée (`gruut 2.2.3`) impose `numpy < 2.0.0` et n’expose pas de build compatible avec `numpy 2.x`.
- **Existe-t-il une version gruut compatible avec numpy 2.2.6 ?** Non à date : il faut soit **revenir à numpy 1.22.x** (ou
  toute version <2.0) pour satisfaire `gruut`, soit isoler les besoins numpy 2.x dans **un autre environnement**.
- **TTS compatible avec numpy/pandas récents ?** `tts 0.22.0` requiert `numpy==1.22.0` (Python ≤3.10) et `pandas<2.0`. Aucune
  roue actuelle ne prend en charge `numpy 2.2.6` ou `pandas 2.3.x`. Conservez donc le couple `numpy 1.22.x` / `pandas <2.0`
  pour les étapes TTS/gruut, et utilisez un environnement séparé si d’autres dépendances exigent des versions plus récentes.

### Dépendances par étape et stratégies d’environnement

| Étape | Bibliothèques clés | Points d’attention |
| --- | --- | --- |
| 01 Extraction audio | `ffmpeg` (CLI) | Disponible via le système ou conda (`conda install -c conda-forge ffmpeg`). |
| 02 Séparation stems | `demucs` (→ `torch`, `torchaudio`), option `uvr` | Aligner `torch/torchaudio/torchvision` sur la même build CUDA (ex. cu121/cu128). Fallback CPU possible, `--demucs-require-cuda` pour forcer l’échec si le GPU n’est pas accessible. |
| 03 Diarisation | `pyannote.audio` (→ `torch`), token HF requis | Nécessite la même pile torch que Demucs pour éviter des conflits (versions, CUDA). |
| 04 Transcription | `faster-whisper` (→ `ctranslate2`, `tokenizers`) | Indépendant de torch, mais gourmand en RAM/GPU ; compatible avec numpy récent. |
| 05 Traduction | `transformers`, `torch` | Même contrainte torch/CUDA que les étapes 02/03 pour mutualiser les GPU. |
| 06 Banque de voix | `pyannote.audio`, `numpy`, `soundfile` | Hérite des contraintes torch + HF token. |
| 07 Attribution personnages | `pyannote.audio`, `numpy`, `soundfile`, `yaml` | Même pile torch/numpy que 06. |
| 08 Synthèse XTTS | `TTS` (Coqui), `gruut`, `numpy==1.22.x`, `pandas<2.0`, `soundfile` | **Incompatible avec numpy/pandas ≥2.x** ; à isoler si d’autres tâches requièrent numpy/pandas récents. |
| 09 Mix audio | `ffmpeg` (CLI) | Pas de dépendance Python supplémentaire. |
| 10 Remux | `ffmpeg` (CLI) | Idem. |

**Quand utiliser plusieurs environnements ?**

- **Environnement unique (simple)** : fonctionne si vous acceptez de rester sur `numpy 1.22.x` / `pandas<2.0` pour satisfaire `TTS/gruut`. Installez torch/torchaudio/torchvision CUDA, demucs, pyannote, transformers, faster-whisper et TTS dans le même env (Python 3.10) puis vérifiez avec `pip check`.
- **Environnements séparés (recommandé si numpy/pandas 2.x sont nécessaires ailleurs)** :
  - `anime_dub_core` : torch CUDA + demucs + pyannote + faster-whisper + transformers (libres vis-à-vis de numpy 2.x).
  - `anime_dub_tts` : `TTS==0.22.0`, `gruut==2.2.3`, `numpy==1.22.0`, `pandas>=1.4,<2.0`, `soundfile` ; pas de dépendance torch nécessaire ici.

**Exécution dans des environnements différents**

1) Créez les environnements :

```bash
# Core (GPU + numpy/pandas récents possibles)
conda create -n anime_dub_core python=3.10
conda activate anime_dub_core
pip install --upgrade "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install demucs pyannote.audio faster-whisper transformers soundfile

# TTS (contraintes numpy/pandas <2.0)
conda create -n anime_dub_tts python=3.10
conda activate anime_dub_tts
pip install "numpy==1.22.0" "pandas>=1.4,<2.0" "tts==0.22.0" "gruut==2.2.3" soundfile
pip check
```

2) Exécutez les étapes 01→07 et 09→10 dans `anime_dub_core` (GUI ou CLI) :

```bash
conda activate anime_dub_core
python scripts/01_extract_audio.py --stem <episode>
...
python scripts/07_assign_characters.py --stem <episode>
python scripts/09_mix_audio.py --stem <episode>
python scripts/10_remux.py --stem <episode>
```

3) Exécutez l’étape 08 dans `anime_dub_tts` en ciblant le même projet/racine (les fichiers d’entrée/sortie restent sous `data/`) :

```bash
conda run -n anime_dub_tts \
  ANIME_DUB_PROJECT_ROOT="<chemin_du_projet>" ANIME_DUB_CONFIG_DIR="<chemin_du_projet>/config" \
  python scripts/08_synthesize_xtts.py --stem <episode> --verbose
```

4) Via le GUI :
   - ouvrez **Options → Configurer les environnements…**. La fenêtre affiche des listes déroulantes alimentées par `conda env list --json` :
     - **Environnement par défaut** (prérempli avec `anime_dub`) appliqué à toutes les étapes sans override.
     - **Étape 03 – Diarisation** (préremplie avec `anime_dub_diar`).
     - **Étape 08 – XTTS** (préremplie avec `anime_dub_tts`).
   - choisissez les environnements dans les listes (ou laissez vide pour utiliser l’environnement courant), saisissez le binaire `conda`/`mamba` si besoin, puis cliquez sur **Enregistrer**. Le GUI utilisera alors automatiquement `conda run -n <env> python -u ...` pour l’étape correspondante, l’environnement par défaut étant appliqué aux autres étapes.

Ainsi, les parties dépendantes de torch/CUDA et les parties contraintes par `TTS/gruut` sont isolées. Vérifiez que les deux environnements pointent vers la même racine de projet pour partager les artefacts, et relancez `pip check` après toute mise à jour majeure.


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
