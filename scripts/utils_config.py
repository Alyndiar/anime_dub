		  # scripts/utils_config.py
"""
Utilitaires pour charger les fichiers de configuration YAML du projet.

- paths.yaml       : chemins des différents dossiers de données
- characters.yaml  : définition des personnages et paramètres de matching
- xtts_config.yaml : configuration du modèle XTTS et des voix de référence
"""

from pathlib import Path
from typing import Any, Dict, Iterable
import yaml


# Racine du projet = dossier parent de "scripts"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(relative_path: str) -> Dict[str, Any]:
    """
    Charge un fichier YAML situé relativement à la racine du projet.

    :param relative_path: Chemin relatif depuis la racine (ex: "config/paths.yaml")
    :return: dictionnaire Python
    """
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Fichier de config introuvable : {path}")
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def load_paths_config() -> Dict[str, Any]:
    """
    Charge config/paths.yaml.

    :return: dict contenant les chemins (ex: "episodes_raw_dir", "audio_raw_dir", etc.)
    """
    return _load_yaml("config/paths.yaml")


def load_characters_config() -> Dict[str, Any]:
    """
    Charge config/characters.yaml.

    Exemple de structure attendue :
    {
        "characters": {
            "char_ning": {...},
            "char_heroine": {...},
            ...
        },
        "matching": {
            "similarity_threshold": 0.7,
            "fallback_character": "char_other"
        }
    }
    """
    return _load_yaml("config/characters.yaml")


def load_xtts_config() -> Dict[str, Any]:
    """
    Charge config/xtts_config.yaml.

    Exemple de structure attendue :
    {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "device": "cuda",
        "sample_rate": 24000,
        "reference_voices": {
            "char_ning": "data/voices/char_ning/ref1.wav",
            ...
        },
        "voice_gain_db": 0.0
    }
    """
    return _load_yaml("config/xtts_config.yaml")


def get_data_path(key: str) -> Path:
    """
    Raccourci pratique pour obtenir un Path vers un répertoire de data défini
    dans config/paths.yaml.

    :param key: nom de la clef dans paths.yaml (ex: "episodes_raw_dir")
    :return: pathlib.Path correspondant
    """
    paths_cfg = load_paths_config()
    if key not in paths_cfg:
        raise KeyError(f"Clé '{key}' absente de paths.yaml")
    return PROJECT_ROOT / paths_cfg[key]


def get_path_map(keys: Iterable[str]) -> Dict[str, Path]:
    """
    Renvoie un dictionnaire {clef: Path} pour une sélection de clefs
    définies dans ``config/paths.yaml``.

    :param keys: liste de clefs à résoudre (ex: ["audio_raw_dir", "diarization_dir"]).
    :return: dictionnaire clef -> pathlib.Path
    """
    return {key: get_data_path(key) for key in keys}


def ensure_directories(keys: Iterable[str]) -> Dict[str, Path]:
    """
    Crée (si nécessaire) les répertoires correspondant aux clefs fournies
    et renvoie le mapping des chemins résolus. Utile pour initialiser la
    structure attendue avant l'exécution d'un script ou d'une future GUI.

    :param keys: clefs de ``config/paths.yaml`` à créer si absentes.
    :return: dictionnaire clef -> pathlib.Path existants/garantis.
    """
    paths = get_path_map(keys)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
