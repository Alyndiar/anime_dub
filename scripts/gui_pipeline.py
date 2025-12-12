"""
Interface GUI pour orchestrer les étapes du pipeline.

Fonctionnalités principales :
- Sélectionner les étapes à exécuter et l'ordre par défaut 01→10.
- Choisir le niveau de pause (aucune, après chaque fichier, répertoire ou étape).
- Modifier les fichiers de configuration (chemins de données) directement depuis l'interface.
- Enregistrer / charger l'état complet : thème, chemins, options, étape + fichier en cours.
- Relancer automatiquement les scripts concernés avec filtrage ``--stem`` pour reprendre sur un fichier précis.

Le GUI est basé sur Tkinter afin d'éviter de nouvelles dépendances.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import yaml

from utils_config import PROJECT_ROOT, load_paths_config


STATE_PATH = PROJECT_ROOT / "config/gui_state.json"


@dataclass
class WorkflowStep:
    step_id: str
    label: str
    script: str
    source_key: str | None
    glob_pattern: str | None
    description: str
    supports_verbose: bool = False

    def list_units(self, path_manager: "PathManager") -> List[str]:
        """Retourne les stems attendus pour cette étape."""
        if not self.source_key or not self.glob_pattern:
            return ["_all_"]
        src_dir = path_manager.get_path(self.source_key)
        stems: list[str] = []
        for path in sorted(src_dir.glob(self.glob_pattern)):
            stem = path.stem
            # Normalise les stems selon les suffixes utilisés dans le pipeline
            stem = stem.replace("_mono16k", "").replace("_fr", "").replace("_segments", "").replace("_fr_voices", "").replace("_fr_full", "")
            stems.append(stem)
        return stems or ["_all_"]


STEPS: list[WorkflowStep] = [
    WorkflowStep("01", "Extraction audio", "01_extract_audio.py", "episodes_raw_dir", "*.mkv", "ffmpeg → wav", supports_verbose=True),
    WorkflowStep("02", "Séparation stems", "02_separate_stems.py", "audio_raw_dir", "*_full.wav", "Demucs/UVR", supports_verbose=True),
    WorkflowStep("03", "Diarisation", "03_diarize.py", "audio_raw_dir", "*_mono16k.wav", "pyannote 3.1"),
    WorkflowStep("04", "Transcription Whisper", "04_whisper_transcribe.py", "audio_raw_dir", "*_mono16k.wav", "Whisper large-v3"),
    WorkflowStep("05", "Traduction NLLB", "05_translate_nllb.py", "whisper_json_dir", "*.json", "NLLB 600M"),
    WorkflowStep("06", "Banque de voix", "06_build_speaker_bank.py", None, None, "Initialisation/embeddings"),
    WorkflowStep("07", "Attribution personnages", "07_assign_characters.py", "whisper_json_fr_dir", "*_fr.json", "Matching embeddings"),
    WorkflowStep("08", "Synthèse XTTS", "08_synthesize_xtts.py", "segments_dir", "*_segments.json", "XTTS-v2"),
    WorkflowStep("09", "Mix audio", "09_mix_audio.py", "dub_audio_dir", "*_fr_voices.wav", "amix ffmpeg"),
    WorkflowStep("10", "Remux vidéo", "10_remux.py", "dub_audio_dir", "*_fr_full.wav", "Remux MKV"),
]


class PathManager:
    """Gère les chemins (base + overrides) et la sauvegarde YAML."""

    def __init__(self, base_dir: Path, overrides: Dict[str, str] | None = None, config_dir: Path | None = None):
        self.base_dir = base_dir
        self.config_dir = config_dir or (PROJECT_ROOT / "config")
        self.overrides = overrides or {}
        self.paths_yaml = self._load_paths_yaml()
        self.parent_map, self.children_map, self.depth_map = self._build_hierarchy()

    def _load_paths_yaml(self) -> Dict[str, str]:
        """Charge le paths.yaml du projet courant ou celui du template racine."""
        candidate = self.config_dir / "paths.yaml"
        if candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        return load_paths_config()

    def _build_hierarchy(self) -> Tuple[Dict[str, str], Dict[str, list[str]], Dict[str, int]]:
        parent_map: Dict[str, str] = {}
        children_map: Dict[str, list[str]] = {}
        depth_map: Dict[str, int] = {}
        items = {k: Path(v) for k, v in self.paths_yaml.items()}

        for key, path in items.items():
            best_parent = None
            best_depth = -1
            for candidate, candidate_path in items.items():
                if candidate == key:
                    continue
                try:
                    path.relative_to(candidate_path)
                except ValueError:
                    continue
                if len(candidate_path.parts) > best_depth:
                    best_parent = candidate
                    best_depth = len(candidate_path.parts)
            if best_parent:
                parent_map[key] = best_parent
                children_map.setdefault(best_parent, []).append(key)
                depth_map[key] = best_depth + 1
            else:
                depth_map[key] = len(Path(self.paths_yaml[key]).parts)
        return parent_map, children_map, depth_map

    def as_display_value(self, key: str) -> str:
        if key in self.overrides:
            return self.overrides[key]
        return self.paths_yaml.get(key, "")

    def _resolve_path(self, key: str, cache: Dict[str, Path]) -> Path:
        if key in cache:
            return cache[key]
        parent_key = self.parent_map.get(key)
        parent_path = self.base_dir if not parent_key else self._resolve_path(parent_key, cache)
        raw_value = self.as_display_value(key)
        raw_path = Path(raw_value)
        if raw_path.is_absolute():
            cache[key] = raw_path
            return raw_path
        parent_parts = Path(self.as_display_value(parent_key)).parts if parent_key else ()
        raw_parts = raw_path.parts
        if parent_parts and len(raw_parts) >= len(parent_parts) and list(raw_parts[: len(parent_parts)]) == list(parent_parts):
            suffix = Path(*raw_parts[len(parent_parts) :])
            resolved = parent_path / suffix
        else:
            resolved = parent_path / raw_path
        cache[key] = resolved
        return resolved

    def get_path(self, key: str) -> Path:
        cache: Dict[str, Path] = {}
        return self._resolve_path(key, cache)

    def update_value(self, key: str, value: str) -> None:
        self.overrides[key] = value

    def save_to_yaml(self) -> None:
        updated = dict(self.paths_yaml)
        updated.update(self.overrides)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        (self.config_dir / "paths.yaml").write_text(
            yaml.safe_dump(updated, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def to_state(self) -> dict:
        return {"base_dir": str(self.base_dir), "overrides": self.overrides, "config_dir": str(self.config_dir)}


class WorkflowState:
    def __init__(self):
        self.theme = "dark"
        self.pause_mode = "step"  # values: none, file, directory, step
        self.selected_steps = [s.step_id for s in STEPS]
        self.current_step = None
        self.current_index = 0
        self.path_state = {"base_dir": str(PROJECT_ROOT), "overrides": {}, "config_dir": str(PROJECT_ROOT / "config")}
        self.project = {"name": None, "anime_title": None, "base_dir": str(PROJECT_ROOT)}
        self.selection_mode = "all"  # values: all, single, selection
        self.single_stem = None
        self.selected_units: list[str] = []
        self.verbose = False
        self.default_env_name: str | None = "anime_dub"
        self.diar_env_name: str | None = "anime_dub_diar"
        self.tts_env_name: str | None = "anime_dub_tts"
        self.conda_command: str = "conda"

    def load(self, path: Path = STATE_PATH) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.theme = data.get("theme", self.theme)
        self.pause_mode = data.get("pause_mode", self.pause_mode)
        self.selected_steps = data.get("selected_steps", self.selected_steps)
        self.current_step = data.get("current_step")
        self.current_index = int(data.get("current_index", 0))
        self.path_state = data.get("paths", self.path_state)
        self.project = data.get("project", self.project)
        self.selection_mode = data.get("selection_mode", self.selection_mode)
        self.single_stem = data.get("single_stem")
        self.selected_units = data.get("selected_units", self.selected_units)
        self.verbose = bool(data.get("verbose", self.verbose))
        self.default_env_name = data.get("default_env_name", self.default_env_name)
        self.diar_env_name = data.get("diar_env_name", self.diar_env_name)
        self.tts_env_name = data.get("tts_env_name", self.tts_env_name)
        self.conda_command = data.get("conda_command", self.conda_command)

    def save(self, path: Path = STATE_PATH) -> None:
        payload = {
            "theme": self.theme,
            "pause_mode": self.pause_mode,
            "selected_steps": self.selected_steps,
            "current_step": self.current_step,
            "current_index": self.current_index,
            "paths": self.path_state,
            "project": self.project,
            "selection_mode": self.selection_mode,
            "single_stem": self.single_stem,
            "selected_units": self.selected_units,
            "verbose": self.verbose,
            "default_env_name": self.default_env_name,
            "diar_env_name": self.diar_env_name,
            "tts_env_name": self.tts_env_name,
            "conda_command": self.conda_command,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class WorkflowRunner:
    """Exécution séquentielle des scripts dans un thread séparé."""

    def __init__(self, log_fn, state: WorkflowState, path_manager: PathManager):
        self.log = log_fn
        self.state = state
        self.path_manager = path_manager
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

    @staticmethod
    def _format_cmd(cmd: list[str]) -> str:
        """Formate une commande pour l'affichage et l'exécution avec shell=True."""
        if os.name == "nt":
            return subprocess.list2cmdline(cmd)
        return shlex.join(cmd)

    def _resolve_env_for_step(self, step_id: str) -> tuple[str | None, str]:
        """Retourne l'environnement conda à utiliser et sa provenance.

        La provenance est l'une de:
        - "diar" : environnement spécifique diarisation renseigné
        - "tts" : environnement spécifique TTS renseigné
        - "default" : environnement par défaut du pipeline
        - "none" : aucun environnement défini, on utilise l'environnement courant
        """

        if step_id == "03" and self.state.diar_env_name:
            return self.state.diar_env_name, "diar"
        if step_id == "08" and self.state.tts_env_name:
            return self.state.tts_env_name, "tts"
        if self.state.default_env_name:
            return self.state.default_env_name, "default"
        return None, "none"

    def _filter_units(self, units: list[str]) -> list[str]:
        if not units:
            return []
        if self.state.selection_mode == "single" and self.state.single_stem:
            filtered = [u for u in units if u == self.state.single_stem]
            # Si le stem ciblé n'est pas présent dans la liste détectée (ex. épisode hors répertoire),
            # on force quand même l'exécution sur ce stem explicitement demandé.
            return filtered or [self.state.single_stem]
        if self.state.selection_mode == "selection" and self.state.selected_units:
            filtered = [u for u in units if u in self.state.selected_units]
            return filtered or list(self.state.selected_units)
        return units

    def start(self, selected_steps: list[str], pause_mode: str):
        if self.thread and self.thread.is_alive():
            self.log("Une exécution est déjà en cours.")
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.state.pause_mode = pause_mode
        self.state.selected_steps = selected_steps
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        if not self.state.selected_steps:
            self.log("Aucune étape sélectionnée.")
            return

        if self.state.pause_mode == "file":
            self._run_by_file()
        else:
            self._run_by_step()

    def _run_by_file(self):
        selected_steps = [s for s in STEPS if s.step_id in self.state.selected_steps]
        if not selected_steps:
            self.log("Aucune étape sélectionnée.")
            return

        units = self._gather_units(selected_steps)
        if not units:
            self.log("Aucun fichier sélectionné pour les étapes demandées.")
            return

        start_index = self.state.current_index if self.state.pause_mode == "file" else 0

        for unit_idx, unit in enumerate(units):
            if unit_idx < start_index:
                continue
            for step in selected_steps:
                if self.stop_event.is_set():
                    self.log("Arrêt demandé, sauvegarde de l'état…")
                    self.state.current_step = step.step_id
                    self.state.current_index = unit_idx
                    self.state.save()
                    return

                units_for_step = self._filter_units(step.list_units(self.path_manager))
                if self.state.verbose:
                    self.log(f"[verbose] Ciblage pour {step.label}: {units_for_step}")
                if not units_for_step:
                    self.log(f"Aucun fichier disponible pour {step.label}, étape ignorée.")
                    continue

                if units_for_step == ["_all_"]:
                    if unit_idx > 0:
                        continue
                    batch = ["_all_"]
                elif unit not in units_for_step:
                    self.log(f"{unit} n'est pas disponible pour {step.label}, étape ignorée pour ce fichier.")
                    continue
                else:
                    batch = [unit]

                self.state.current_step = step.step_id
                self.state.current_index = unit_idx
                self.state.save()

                label_units = ", ".join(batch)
                self.log(f"→ {step.label} : {label_units}")
                self._run_script(step, batch)

                if self.pause_event.is_set():
                    self.log("Mise en pause demandée.")
                    self.state.save()
                    return

            if self.state.pause_mode == "file" and unit != "_all_":
                self.log("Pause après fichier (option active).")
                self.state.current_step = None
                self.state.current_index = unit_idx + 1
                self.state.save()
                self.pause_event.set()
                return

        self.log("Pipeline terminé.")
        self.state.current_step = None
        self.state.current_index = 0
        self.state.save()

    def _next_step_id(self, current: str) -> str | None:
        ordered = [s.step_id for s in STEPS if s.step_id in self.state.selected_steps]
        if current not in ordered:
            return None
        idx = ordered.index(current)
        return ordered[idx + 1] if idx + 1 < len(ordered) else None

    def _run_by_step(self):
        start_step = self.state.current_step or (self.state.selected_steps[0] if self.state.selected_steps else None)
        start_found = False if start_step else True

        for step in STEPS:
            if step.step_id not in self.state.selected_steps:
                continue
            if not start_found:
                if step.step_id == start_step:
                    start_found = True
                else:
                    continue

            units = self._filter_units(step.list_units(self.path_manager))
            if self.state.verbose:
                self.log(f"[verbose] Fichiers pour {step.label}: {units}")
            if not units:
                self.log(f"Aucun fichier sélectionné pour {step.label}, étape ignorée.")
                continue
            start_index = self.state.current_index if self.state.current_step == step.step_id else 0
            for batch_idx, batch in enumerate([units]):
                if batch_idx < start_index:
                    continue
                if self.stop_event.is_set():
                    self.log("Arrêt demandé, sauvegarde de l'état…")
                    self.state.current_step = step.step_id
                    self.state.current_index = batch_idx
                    self.state.save()
                    return

                self.state.current_step = step.step_id
                self.state.current_index = batch_idx
                self.state.save()

                label_units = ", ".join(batch)
                self.log(f"→ {step.label} : {label_units}")
                self._run_script(step, batch)

                if self.pause_event.is_set():
                    self.log("Mise en pause demandée.")
                    self.state.save()
                    return

            if self.state.pause_mode == "step":
                next_step = self._next_step_id(step.step_id)
                if next_step is None:
                    continue
                self.log("Pause après étape (option active).")
                self.state.current_step = next_step
                self.state.current_index = 0
                self.state.save()
                self.pause_event.set()
                return

        if self.state.pause_mode == "directory":
            self.log("Pipeline terminé pour ce répertoire.")
            self.log("Pause après répertoire (option active).")
            self.state.current_step = None
            self.state.current_index = 0
            self.state.save()
            self.pause_event.set()
            return

        self.log("Pipeline terminé.")
        self.state.current_step = None
        self.state.current_index = 0
        self.state.save()

    def _gather_units(self, selected_steps: list[WorkflowStep]) -> list[str]:
        for step in selected_steps:
            units = self._filter_units(step.list_units(self.path_manager))
            candidates = [u for u in units if u != "_all_"]
            if candidates:
                return candidates
        return []

    def _run_script(self, step: WorkflowStep, units: list[str]):
        script_path = PROJECT_ROOT / "scripts" / step.script
        env_name, env_source = self._resolve_env_for_step(step.step_id)
        conda_cmd = self.state.conda_command or "conda"
        if env_name:
            cmd = [
                conda_cmd,
                "run",
                "-n",
                env_name,
                "python",
                "-u",
                str(script_path),
            ]
            use_shell = True
            if step.step_id == "03":
                if env_source == "diar":
                    self.log(f"[info] Étape 03 (Diarisation) exécutée via {conda_cmd} run -n {env_name}")
                else:
                    self.log(
                        f"[warn] Aucun environnement dédié diarisation renseigné, utilisation de {env_name} (source {env_source})"
                    )
            elif step.step_id == "08":
                if env_source == "tts":
                    self.log(f"[info] Étape 08 (XTTS) exécutée via {conda_cmd} run -n {env_name}")
                else:
                    self.log(
                        f"[warn] Aucun environnement dédié XTTS renseigné, utilisation de {env_name} (source {env_source})"
                    )
            elif env_source == "default":
                self.log(
                    f"[info] {step.label} exécuté via {conda_cmd} run -n {env_name} (environnement par défaut)"
                )
        else:
            cmd = ["python", "-u", str(script_path)]
            use_shell = False
        stems = [u for u in units if u != "_all_"]
        for stem in stems:
            cmd.extend(["--stem", stem])
        if self.state.verbose and step.supports_verbose:
            cmd.append("--verbose")
        env = os.environ.copy()
        env["ANIME_DUB_PROJECT_ROOT"] = str(self.path_manager.base_dir)
        env["ANIME_DUB_CONFIG_DIR"] = str(self.path_manager.config_dir)
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.state.verbose:
            env["ANIME_DUB_VERBOSE"] = "1"
        if stems:
            env["ANIME_DUB_SELECTED_STEMS"] = json.dumps(stems, ensure_ascii=False)
        env_summary = {k: v for k, v in env.items() if k.startswith("ANIME_DUB_")}
        formatted_cmd = self._format_cmd(cmd) if use_shell else " ".join(cmd)
        self.log(f"Exécution de {formatted_cmd}.")
        if self.state.verbose:
            self.log(f"[verbose] Environnement : {env_summary}")
        try:
            process = subprocess.Popen(
                formatted_cmd if use_shell else cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
                universal_newlines=True,
                shell=use_shell,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.log(line.rstrip())
            return_code = process.wait()
            if return_code != 0:
                self.log(f"Erreur lors de {step.label} : exit code {return_code}")
                self.pause_event.set()
                self.state.save()
        except OSError as exc:
            self.log(f"Erreur lors de {step.label} : {exc}")
            self.pause_event.set()
            self.state.save()

    def stop(self):
        self.stop_event.set()

    def pause(self):
        self.pause_event.set()


class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = WorkflowState()
        self.state.load()

        self.project_name = None
        self.anime_title = None
        self.project_base = Path(self.state.project.get("base_dir", PROJECT_ROOT))
        self.project_config_dir = self.project_base / "config"
        self._maybe_load_project_state()

        base_dir = Path(self.state.path_state.get("base_dir", self.project_base))
        overrides = self.state.path_state.get("overrides", {})
        config_dir = Path(self.state.path_state.get("config_dir", self.project_config_dir))
        self.path_manager = PathManager(base_dir=base_dir, overrides=overrides, config_dir=config_dir)
        self.log_lock = threading.Lock()
        self.log_file = self._compute_log_path()

        self.runner = WorkflowRunner(self.log, self.state, self.path_manager)
        self.step_vars: Dict[str, tk.BooleanVar] = {}
        self.path_vars: Dict[str, tk.StringVar] = {}
        self.path_last_values: Dict[str, str] = {}
        self.resolved_labels: Dict[str, ttk.Label] = {}
        self.base_var = tk.StringVar(value=str(self.path_manager.base_dir))
        self.base_trace_added = False
        self.dialogs: set[tk.Toplevel] = set()
        self.selection_mode_var = tk.StringVar(value=self.state.selection_mode)
        self.single_stem_var = tk.StringVar(value=self.state.single_stem or "")
        self.selected_units: list[str] = list(self.state.selected_units)
        self.available_stems: list[str] = []
        self.verbose_var = tk.BooleanVar(value=self.state.verbose)
        self.available_envs: list[str] = []

        self._build_menu()
        self._build_layout()
        self._update_title()
        self._update_log_destination()
        self.apply_theme(self.state.theme)

    def _maybe_load_project_state(self):
        project_base = Path(self.state.project.get("base_dir", PROJECT_ROOT)) if self.state.project else PROJECT_ROOT
        metadata_path = project_base / "config/gui_state.json"
        if metadata_path.exists():
            self.state.load(metadata_path)
        self.project_name = self.state.project.get("name") if self.state.project else None
        self.anime_title = self.state.project.get("anime_title") if self.state.project else None
        self.project_base = Path(self.state.project.get("base_dir", PROJECT_ROOT))
        self.project_config_dir = self.project_base / "config"

    def _build_menu(self):
        self.menubar = tk.Menu(self.root)

        self.project_menu = tk.Menu(self.menubar, tearoff=0)
        self.project_menu.add_command(label="Créer un projet", command=self.create_project)
        self.project_menu.add_command(label="Charger un projet", command=self.load_project)
        self.project_menu.add_command(label="Sauvegarder le projet", command=self.save_project)
        self.project_menu.add_command(label="Fermer le projet", command=self.close_project)
        self.menubar.add_cascade(label="Projet", menu=self.project_menu)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Enregistrer l'état", command=self.save_state)
        self.file_menu.add_command(label="Charger un état", command=self.load_state)
        self.file_menu.add_command(label="Enregistrer sous…", command=self.save_state_as)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Quitter", command=self.root.destroy)
        self.menubar.add_cascade(label="Fichier", menu=self.file_menu)

        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.view_menu.add_command(label="Mode Nuit", command=lambda: self.apply_theme("dark"))
        self.view_menu.add_command(label="Mode Jour", command=lambda: self.apply_theme("light"))
        self.menubar.add_cascade(label="Affichage", menu=self.view_menu)

        self.options_menu = tk.Menu(self.menubar, tearoff=0)
        self.options_menu.add_command(label="Configurer les chemins…", command=self.open_paths_window)
        self.options_menu.add_separator()
        self.options_menu.add_checkbutton(
            label="Verbose (traces détaillées)",
            onvalue=True,
            offvalue=False,
            variable=self.verbose_var,
            command=self._toggle_verbose,
        )
        self.options_menu.add_command(label="Configurer les environnements…", command=self.open_envs_window)
        self.options_menu.add_separator()
        pause_menu = tk.Menu(self.options_menu, tearoff=0)
        pause_menu.add_command(label="Aucune pause", command=lambda: self._set_pause("none"))
        pause_menu.add_command(label="Pause par fichier", command=lambda: self._set_pause("file"))
        pause_menu.add_command(label="Pause par répertoire", command=lambda: self._set_pause("directory"))
        pause_menu.add_command(label="Pause par étape", command=lambda: self._set_pause("step"))
        self.options_menu.add_cascade(label="Mode de pause", menu=pause_menu)
        self.menubar.add_cascade(label="Options", menu=self.options_menu)

        self.root.config(menu=self.menubar)

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=10, style="Bg.TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        # Section étapes
        steps_frame = ttk.LabelFrame(container, text="Étapes du workflow", style="Card.TLabelframe")
        steps_frame.pack(fill=tk.X, expand=False, pady=5)

        for idx, step in enumerate(STEPS):
            var = tk.BooleanVar(value=step.step_id in self.state.selected_steps)
            self.step_vars[step.step_id] = var
            ttk.Checkbutton(
                steps_frame,
                text=f"{step.step_id} – {step.label} ({step.description})",
                variable=var,
                style="Card.TCheckbutton",
            ).grid(row=idx, column=0, sticky="w")

        # Section sélection d'épisodes
        self._refresh_available_stems()
        selection_frame = ttk.LabelFrame(container, text="Ciblage des épisodes", style="Card.TLabelframe")
        selection_frame.pack(fill=tk.X, expand=False, pady=5)

        ttk.Radiobutton(
            selection_frame,
            text="Tous les épisodes", 
            variable=self.selection_mode_var,
            value="all",
            command=self._on_selection_mode_change,
            style="Card.TRadiobutton",
        ).grid(row=0, column=0, sticky="w")

        ttk.Radiobutton(
            selection_frame,
            text="1 seul épisode",
            variable=self.selection_mode_var,
            value="single",
            command=self._on_selection_mode_change,
            style="Card.TRadiobutton",
        ).grid(row=1, column=0, sticky="w")
        single_container = ttk.Frame(selection_frame, style="Bg.TFrame")
        single_container.grid(row=1, column=1, columnspan=2, sticky="w")
        self.single_combo = ttk.Combobox(single_container, textvariable=self.single_stem_var, values=self.available_stems, width=40)
        self.single_combo.grid(row=0, column=0, padx=(6, 4), sticky="w")
        self.single_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_single_change())
        ttk.Button(single_container, text="Parcourir…", command=self._choose_single_unit_from_file, style="Accent.TButton").grid(row=0, column=1, padx=(0, 4))
        ttk.Button(single_container, text="Rafraîchir", command=self._refresh_available_stems, style="Accent.TButton").grid(row=0, column=2)

        ttk.Radiobutton(
            selection_frame,
            text="Sélection par fichiers…",
            variable=self.selection_mode_var,
            value="selection",
            command=self._on_selection_mode_change,
            style="Card.TRadiobutton",
        ).grid(row=2, column=0, sticky="w")
        ttk.Button(selection_frame, text="Choisir des fichiers", command=self._choose_units_from_files, style="Accent.TButton").grid(row=2, column=1, sticky="w", padx=6)
        self.selection_label = ttk.Label(selection_frame, text=self._selection_summary())
        self.selection_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        selection_frame.columnconfigure(1, weight=1)

        # Section commandes
        controls = ttk.Frame(container, style="Bg.TFrame")
        controls.pack(fill=tk.X, pady=5)

        ttk.Button(controls, text="Démarrer / Reprendre", command=self.start_workflow, style="Accent.TButton").pack(side=tk.LEFT)
        ttk.Button(controls, text="Pause", command=self.runner.pause, style="Accent.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Arrêter", command=self.runner.stop, style="Accent.TButton").pack(side=tk.LEFT)

        ttk.Checkbutton(controls, text="Verbose", variable=self.verbose_var, style="Card.TCheckbutton", command=self._toggle_verbose).pack(side=tk.LEFT, padx=(16, 4))

        ttk.Label(controls, text="Mode de pause :").pack(side=tk.LEFT, padx=(20, 4))
        self.pause_var = tk.StringVar(value=self.state.pause_mode)
        pause_combo = ttk.Combobox(controls, textvariable=self.pause_var, values=["none", "file", "directory", "step"], width=12)
        pause_combo.pack(side=tk.LEFT)

        # Log
        log_frame = ttk.LabelFrame(container, text="Logs", style="Card.TLabelframe")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_widget = tk.Text(log_frame, height=15, wrap="word")
        self.log_widget.pack(fill=tk.BOTH, expand=True)
        self.env_list_warning_shown = False

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.log_widget.insert(tk.END, formatted + "\n")
        self.log_widget.see(tk.END)
        try:
            with self.log_lock:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                self.log_file.open("a", encoding="utf-8").write(formatted + "\n")
        except OSError:
            pass

    def _compute_log_path(self) -> Path:
        base = getattr(self, "path_manager", None)
        if base:
            return base.base_dir / "logs" / "gui_pipeline.log"
        return PROJECT_ROOT / "logs" / "gui_pipeline.log"

    def _update_log_destination(self):
        self.log_file = self._compute_log_path()

    def _normalize_stem(self, value: str) -> str:
        stem = Path(value).stem
        return (
            stem.replace("_mono16k", "")
            .replace("_fr", "")
            .replace("_segments", "")
            .replace("_fr_voices", "")
            .replace("_fr_full", "")
        )

    def _refresh_available_stems(self):
        candidate_step = STEPS[0]
        stems = [u for u in candidate_step.list_units(self.path_manager) if u != "_all_"]
        stems = [self._normalize_stem(s) for s in stems]
        stems = sorted(set(stems))
        self.available_stems = stems
        if hasattr(self, "single_combo"):
            self.single_combo.configure(values=self.available_stems)
        if stems and not self.single_stem_var.get():
            self.single_stem_var.set(stems[0])
        if hasattr(self, "selection_label"):
            self.selection_label.configure(text=self._selection_summary())

    def _list_conda_envs(self) -> list[str]:
        executable = self.state.conda_command or "conda"
        cmd = [executable, "env", "list", "--json"]
        cmd_str = WorkflowRunner._format_cmd(cmd)
        try:
            self.log(f"[verbose] Exécution de {cmd_str}")
            output = subprocess.check_output(cmd_str, text=True, shell=True)
            data = json.loads(output)
            envs = data.get("envs", [])
            names = sorted({Path(path).name for path in envs})
            return names
        except FileNotFoundError as exc:  # pragma: no cover - dépend du système de l'utilisateur
            if not self.env_list_warning_shown:
                self.log(
                    "[warn] Impossible de lister les environnements conda (commande introuvable). "
                    "Vérifie que conda/mamba est dans le PATH ou change la commande via la fenêtre ‘Configurer les environnements…’."
                )
                self.env_list_warning_shown = True
            else:
                self.log(f"[warn] Impossible de lister les environnements conda ({exc})")
            return []
        except Exception as exc:  # pragma: no cover - dépend du système de l'utilisateur
            self.log(f"[warn] Impossible de lister les environnements conda ({exc})")
            return []

    def _env_options(self, current: str | None) -> list[str]:
        options = list(self.available_envs)
        if current and current not in options:
            options.insert(0, current)
        return options

    def _refresh_env_choices(self):
        self.available_envs = self._list_conda_envs()
        for key, combo in getattr(self, "env_combos", {}).items():
            current = self.env_vars[key].get()
            combo.configure(values=self._env_options(current))
        self._update_conda_radio_states()

    def _update_conda_radio_states(self):
        conda_path = shutil.which("conda")
        mamba_path = shutil.which("mamba")
        if hasattr(self, "conda_radio") and self.conda_radio:
            if conda_path:
                self.conda_radio.state(["!disabled"])
            else:
                self.conda_radio.state(["disabled"])
        if hasattr(self, "mamba_radio") and self.mamba_radio:
            if mamba_path:
                self.mamba_radio.state(["!disabled"])
            else:
                self.mamba_radio.state(["disabled"])
        chosen = self.conda_cmd_var.get()
        if chosen == "conda" and not conda_path and mamba_path:
            self.conda_cmd_var.set("mamba")
        elif chosen == "mamba" and not mamba_path and conda_path:
            self.conda_cmd_var.set("conda")
        self.state.conda_command = self.conda_cmd_var.get()

    def open_envs_window(self):
        if hasattr(self, "env_window") and self.env_window.winfo_exists():
            self.env_window.focus_set()
            return

        self.available_envs = self._list_conda_envs()
        self.env_window = tk.Toplevel(self.root)
        self.env_window.title("Environnements conda")
        self.dialogs.add(self.env_window)
        self.env_window.protocol("WM_DELETE_WINDOW", lambda: self._close_dialog(self.env_window))
        container = ttk.Frame(self.env_window, padding=10, style="Bg.TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        info = (
            "Choisissez l'environnement conda par défaut (anime_dub conseillé) et les environnements spécifiques par étape. "
            "Laisser un champ vide exécutera l'étape dans l'environnement courant."
        )
        ttk.Label(container, text=info, wraplength=520).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.env_vars = {
            "default": tk.StringVar(value=self.state.default_env_name or "anime_dub"),
            "diar": tk.StringVar(value=self.state.diar_env_name or "anime_dub_diar"),
            "tts": tk.StringVar(value=self.state.tts_env_name or "anime_dub_tts"),
        }
        self.env_combos: dict[str, ttk.Combobox] = {}

        rows = [
            ("Environnement par défaut", "default"),
            ("Étape 03 – Diarisation", "diar"),
            ("Étape 08 – XTTS", "tts"),
        ]
        for idx, (label, key) in enumerate(rows, start=1):
            ttk.Label(container, text=label).grid(row=idx, column=0, sticky="w", padx=(0, 6), pady=2)
            combo = ttk.Combobox(
                container,
                textvariable=self.env_vars[key],
                values=self._env_options(self.env_vars[key].get()),
                width=30,
            )
            combo.grid(row=idx, column=1, sticky="ew", pady=2)
            self.env_combos[key] = combo

        ttk.Label(container, text="Commande conda/mamba").grid(row=len(rows) + 1, column=0, sticky="w", padx=(0, 6), pady=(10, 2))
        self.conda_cmd_var = tk.StringVar(value=self.state.conda_command or "conda")
        radio_frame = ttk.Frame(container, style="Bg.TFrame")
        radio_frame.grid(row=len(rows) + 1, column=1, sticky="w", pady=(10, 2))
        self.conda_radio = ttk.Radiobutton(
            radio_frame,
            text="conda",
            variable=self.conda_cmd_var,
            value="conda",
            command=self._update_conda_radio_states,
            style="Card.TRadiobutton",
        )
        self.conda_radio.pack(side=tk.LEFT, padx=(0, 8))
        self.mamba_radio = ttk.Radiobutton(
            radio_frame,
            text="mamba",
            variable=self.conda_cmd_var,
            value="mamba",
            command=self._update_conda_radio_states,
            style="Card.TRadiobutton",
        )
        self.mamba_radio.pack(side=tk.LEFT)

        buttons = ttk.Frame(container, style="Bg.TFrame")
        buttons.grid(row=len(rows) + 2, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Rafraîchir la liste", command=self._refresh_env_choices, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Enregistrer", command=self._apply_env_choices, style="Accent.TButton").pack(side=tk.LEFT)

        container.columnconfigure(1, weight=1)
        self._apply_window_background(self.env_window)
        self._update_conda_radio_states()

    def _apply_env_choices(self):
        self.state.default_env_name = self.env_vars["default"].get().strip() or None
        self.state.diar_env_name = self.env_vars["diar"].get().strip() or None
        self.state.tts_env_name = self.env_vars["tts"].get().strip() or None
        self.state.conda_command = self.conda_cmd_var.get() or "conda"
        self._save_state()
        self.log(
            "[info] Environnements : défaut={} | diarisation={} | TTS={} | conda={}".format(
                self.state.default_env_name or "courant",
                self.state.diar_env_name or "courant",
                self.state.tts_env_name or "courant",
                self.state.conda_command,
            )
        )
        if hasattr(self, "env_window") and self.env_window.winfo_exists():
            self.env_window.focus_set()

    def open_paths_window(self):
        if hasattr(self, "paths_window") and self.paths_window.winfo_exists():
            self.paths_window.focus_set()
            return

        self.paths_window = tk.Toplevel(self.root)
        self.paths_window.title("Chemins et répertoires")
        self.dialogs.add(self.paths_window)
        self.paths_window.protocol("WM_DELETE_WINDOW", lambda: self._close_dialog(self.paths_window))
        container = ttk.Frame(self.paths_window, padding=10, style="Bg.TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container, style="Bg.TFrame")
        header.grid(row=0, column=0, columnspan=4, sticky="ew")
        ttk.Label(header, text="Répertoire de base du projet").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(header, textvariable=self.base_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if not self.base_trace_added:
            self.base_var.trace_add("write", lambda *_: self._on_base_change())
            self.base_trace_added = True
        ttk.Button(header, text="...", width=4, command=self._choose_base_dir, style="Accent.TButton").pack(side=tk.LEFT, padx=4)

        headers = ["Clé", "Chemin relatif/absolu", "Résolu", ""]
        for idx, title in enumerate(headers):
            ttk.Label(container, text=title).grid(row=1, column=idx, sticky="w", padx=2)

        self._ensure_path_vars()
        row = 2
        ordered_keys = sorted(self.path_manager.paths_yaml.keys(), key=lambda k: self.path_manager.depth_map.get(k, 0))
        for key in ordered_keys:
            var = self.path_vars[key]
            indent = max(0, self.path_manager.depth_map.get(key, 0) - 1)
            prefix = f"{'    ' * indent}{'↳ ' if indent else ''}"
            label = ttk.Label(container, text=f"{prefix}{key}")
            label.grid(row=row, column=0, sticky="w", padx=(indent * 12, 4))
            entry = ttk.Entry(container, textvariable=var, width=50)
            entry.grid(row=row, column=1, sticky="ew", padx=2)
            entry.bind("<FocusOut>", lambda _evt, k=key: self._on_path_change(k))
            var.trace_add("write", lambda *_args, k=key: self._on_path_change(k))
            browse = ttk.Button(container, text="...", width=4, command=lambda k=key: self._choose_path_dir(k), style="Accent.TButton")
            browse.grid(row=row, column=3, sticky="e", padx=2)
            resolved = ttk.Label(container, text=str(self.path_manager.get_path(key)))
            resolved.grid(row=row, column=2, sticky="w", padx=2)
            self.resolved_labels[key] = resolved
            row += 1

        ttk.Button(container, text="Enregistrer paths.yaml", command=self._save_paths_yaml, style="Accent.TButton").grid(row=row, column=3, pady=6, sticky="e")
        for col in (1, 2):
            container.columnconfigure(col, weight=1)
        self._apply_window_background(self.paths_window)

    def _set_pause(self, mode: str):
        self.pause_var.set(mode)
        self.state.pause_mode = mode
        self._save_state()

    def _toggle_verbose(self):
        self.state.verbose = bool(self.verbose_var.get())
        if self.state.verbose:
            self.log("[verbose] Traces détaillées activées.")
        self._save_state()

    def start_workflow(self):
        selected = [step_id for step_id, var in self.step_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("Étapes", "Sélectionne au moins une étape à exécuter.")
            return
        self.state.selected_steps = selected
        self.state.path_state = self.path_manager.to_state()
        self.state.pause_mode = self.pause_var.get()
        self.state.selection_mode = self.selection_mode_var.get()
        self.state.single_stem = self.single_stem_var.get() or None
        self.state.selected_units = self.selected_units
        self.state.verbose = bool(self.verbose_var.get())
        self._save_state()
        self.runner.start(selected, self.pause_var.get())

    def save_state(self):
        self.state.path_state = self.path_manager.to_state()
        saved_at = self._save_state()
        messagebox.showinfo("État", f"Sauvegarde enregistrée dans {saved_at}")

    def save_state_as(self):
        dest = filedialog.asksaveasfilename(defaultextension=".json", initialdir=str(self._current_state_path().parent))
        if not dest:
            return
        self.state.path_state = self.path_manager.to_state()
        self.state.save(Path(dest))

    def load_state(self):
        src = filedialog.askopenfilename(defaultextension=".json", initialdir=str(self._current_state_path().parent))
        if not src:
            return
        self.state.load(Path(src))
        self.pause_var.set(self.state.pause_mode)
        self.apply_theme(self.state.theme)
        self.verbose_var.set(self.state.verbose)
        self.project_base = Path(self.state.project.get("base_dir", PROJECT_ROOT))
        self.project_name = self.state.project.get("name")
        self.anime_title = self.state.project.get("anime_title")
        self.project_config_dir = Path(self.state.path_state.get("config_dir", self.project_config_dir))
        self.path_manager = PathManager(base_dir=self.project_base, overrides=self.state.path_state.get("overrides", {}), config_dir=self.project_config_dir)
        self._update_log_destination()
        self._refresh_path_vars()
        for step in STEPS:
            self.step_vars[step.step_id].set(step.step_id in self.state.selected_steps)
        self.selection_mode_var.set(self.state.selection_mode)
        self.single_stem_var.set(self.state.single_stem or "")
        self.selected_units = list(self.state.selected_units)
        self._refresh_available_stems()
        if hasattr(self, "selection_label"):
            self.selection_label.configure(text=self._selection_summary())
        self.log("État chargé.")
        self._update_title()

    def _current_state_path(self) -> Path:
        project_path = Path(self.state.project.get("base_dir", "")) if self.state.project else None
        if project_path and project_path != PROJECT_ROOT:
            return project_path / "config/gui_state.json"
        return STATE_PATH

    def _on_selection_mode_change(self):
        mode = self.selection_mode_var.get()
        if mode == "single" and not self.single_stem_var.get() and self.available_stems:
            self.single_stem_var.set(self.available_stems[0])
        self.selection_label.configure(text=self._selection_summary())
        self.state.selection_mode = mode
        self.state.single_stem = self.single_stem_var.get() or None
        self.state.selected_units = self.selected_units
        self._save_state()

    def _selection_summary(self) -> str:
        mode = self.selection_mode_var.get()
        if mode == "all":
            return "Tous les épisodes seront traités."
        if mode == "single":
            return f"Épisode ciblé : {self.single_stem_var.get() or 'aucun'}"
        return f"Épisodes sélectionnés : {', '.join(self.selected_units) if self.selected_units else 'aucun'}"

    def _on_single_change(self):
        self.selection_mode_var.set("single")
        self.selection_label.configure(text=self._selection_summary())
        self.state.single_stem = self.single_stem_var.get() or None
        self.state.selection_mode = "single"
        self._save_state()

    def _choose_units_from_files(self):
        try:
            initial_dir = self.path_manager.get_path("episodes_raw_dir")
        except Exception:
            initial_dir = self.path_manager.base_dir
        filenames = filedialog.askopenfilenames(initialdir=str(initial_dir))
        if not filenames:
            return
        stems = [self._normalize_stem(name) for name in filenames]
        unique = []
        for stem in stems:
            if stem not in unique:
                unique.append(stem)
        self.selected_units = unique
        self.selection_mode_var.set("selection")
        self.selection_label.configure(text=self._selection_summary())
        self.state.selected_units = self.selected_units
        self.state.selection_mode = "selection"
        self._save_state()

    def _choose_single_unit_from_file(self):
        try:
            initial_dir = self.path_manager.get_path("episodes_raw_dir")
        except Exception:
            initial_dir = self.path_manager.base_dir
        filename = filedialog.askopenfilename(initialdir=str(initial_dir))
        if not filename:
            return
        stem = self._normalize_stem(filename)
        if stem not in self.available_stems:
            self.available_stems.append(stem)
            self.available_stems = sorted(set(self.available_stems))
            self.single_combo.configure(values=self.available_stems)
        self.single_stem_var.set(stem)
        self.selection_mode_var.set("single")
        self.selection_label.configure(text=self._selection_summary())
        self.state.single_stem = stem
        self.state.selection_mode = "single"
        self._save_state()

    def _save_state(self) -> Path:
        path = self._current_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.state.save(path)
        # Sauvegarde miroir dans le dossier racine pour recharger le dernier projet ouvert
        if path != STATE_PATH:
            self.state.save(STATE_PATH)
        return path

    # --- Gestion de projet ---
    def create_project(self):
        anime = simpledialog.askstring("Nouvel animé", "Titre de l'animé :", parent=self.root)
        if not anime:
            return
        default_project = anime.strip() or "projet_anime"
        project = simpledialog.askstring(
            "Nom du projet",
            "Nom du projet (sera utilisé pour le dossier et le fichier YAML) :",
            initialvalue=default_project,
            parent=self.root,
        )
        if not project:
            return
        base_path = self._prompt_new_base_dir(project)
        if not base_path:
            return

        try:
            self._initialize_project_files(base_path, anime.strip(), project.strip())
        except FileExistsError:
            messagebox.showerror("Projet", "Le répertoire du projet existe déjà, choisis un chemin inexistant.")
            return

        self.state.project = {"name": project.strip(), "anime_title": anime.strip(), "base_dir": str(base_path)}
        self.project_name = project.strip()
        self.anime_title = anime.strip()
        self.project_base = base_path
        self.project_config_dir = base_path / "config"
        self.path_manager = PathManager(base_dir=base_path, overrides={}, config_dir=self.project_config_dir)
        self._update_log_destination()
        self.state.path_state = self.path_manager.to_state()
        self.base_var.set(str(base_path))
        self._refresh_path_vars(reset=True)
        self._update_title()
        self._save_state()
        self.log(f"Projet créé : {self.project_name} ({base_path})")
        self.open_paths_window()

    def load_project(self):
        base_dir = filedialog.askdirectory(title="Charger un projet (sélectionner le répertoire de base)")
        if not base_dir:
            return
        base_path = Path(base_dir)
        meta_state = base_path / "config/gui_state.json"
        project_name, anime_title = self._load_project_metadata(base_path)
        self.state.project = {"name": project_name, "anime_title": anime_title, "base_dir": str(base_path)}
        self.project_name = project_name
        self.anime_title = anime_title
        self.project_base = base_path
        self.project_config_dir = base_path / "config"
        if meta_state.exists():
            self.state.load(meta_state)
        overrides = self.state.path_state.get("overrides", {})
        config_dir = Path(self.state.path_state.get("config_dir", self.project_config_dir))
        self.path_manager = PathManager(base_dir=base_path, overrides=overrides, config_dir=config_dir)
        self._update_log_destination()
        self.base_var.set(str(base_path))
        self._refresh_path_vars(reset=True)
        self._update_title()
        self._save_state()
        self.log(f"Projet chargé : {self.project_name} ({base_path})")

    def save_project(self):
        if not self.project_base or self.project_base == PROJECT_ROOT:
            messagebox.showwarning("Projet", "Aucun projet dédié n'est ouvert.")
            return
        self.state.project = {"name": self.project_name, "anime_title": self.anime_title, "base_dir": str(self.project_base)}
        self.state.path_state = self.path_manager.to_state()
        self._save_state()
        self._write_project_metadata(self.project_base, self.project_name, self.anime_title)
        meta_json = self.project_base / "config/project.json"
        meta_json.parent.mkdir(parents=True, exist_ok=True)
        meta_json.write_text(
            json.dumps({"name": self.project_name, "anime_title": self.anime_title, "base_dir": str(self.project_base)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        messagebox.showinfo("Projet", f"Projet sauvegardé : {self.project_name}")

    def close_project(self):
        if not self.project_name:
            return
        self.save_project()
        self.project_name = None
        self.anime_title = None
        self.project_base = PROJECT_ROOT
        self.project_config_dir = PROJECT_ROOT / "config"
        self.state.project = {"name": None, "anime_title": None, "base_dir": str(PROJECT_ROOT)}
        self.path_manager = PathManager(base_dir=PROJECT_ROOT, overrides={}, config_dir=self.project_config_dir)
        self._update_log_destination()
        self.state.path_state = self.path_manager.to_state()
        self.base_var.set(str(PROJECT_ROOT))
        self._refresh_path_vars(reset=True)
        self._save_state()
        self.log("Projet fermé, retour au profil par défaut.")
        self._update_title()

    def _update_title(self):
        anime = self.anime_title or "Animé non renseigné"
        project = self.project_name or "Profil par défaut"
        self.root.title(f"Anime Dub – {anime} [{project}]")

    def _prompt_new_base_dir(self, project_name: str) -> Path | None:
        default_base = (PROJECT_ROOT / project_name).resolve()
        current_value = str(default_base)
        while True:
            response = simpledialog.askstring(
                "Répertoire du projet",
                "Choisir un répertoire de base inexistant (sera créé avec la hiérarchie standard) :",
                initialvalue=current_value,
                parent=self.root,
            )
            if response is None:
                return None
            candidate = Path(response).expanduser()
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            if candidate.exists():
                messagebox.showerror("Projet", "Le répertoire existe déjà. Choisis un chemin inexistant.")
                current_value = str(candidate)
                continue
            if candidate.resolve() == PROJECT_ROOT.resolve():
                messagebox.showerror("Projet", "Le répertoire du projet doit être distinct du dépôt.")
                current_value = str(candidate)
                continue
            return candidate

    def _initialize_project_files(self, base_path: Path, anime_title: str, project_name: str) -> None:
        base_path.mkdir(parents=True, exist_ok=False)
        config_dir = base_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        for cfg_name in ("paths.yaml", "characters.yaml", "xtts_config.yaml"):
            template = PROJECT_ROOT / "config" / cfg_name
            dest = config_dir / cfg_name
            if template.exists() and not dest.exists():
                dest.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        path_manager = PathManager(base_dir=base_path, overrides={}, config_dir=config_dir)
        for key in path_manager.paths_yaml:
            try:
                resolved = path_manager.get_path(key)
            except Exception:
                continue
            resolved.mkdir(parents=True, exist_ok=True)
        self._write_project_metadata(base_path, project_name, anime_title)
        meta_json = config_dir / "project.json"
        meta_json.write_text(
            json.dumps({"name": project_name, "anime_title": anime_title, "base_dir": str(base_path)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_project_metadata(self, base_path: Path, project_name: str | None, anime_title: str | None) -> None:
        if not project_name:
            return
        meta = {"project_name": project_name, "anime_title": anime_title, "base_dir": str(base_path)}
        yaml_path = base_path / f"{project_name}.yaml"
        yaml_path.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _load_project_metadata(self, base_path: Path) -> Tuple[str | None, str | None]:
        primary_yaml = base_path / f"{base_path.name}.yaml"
        if primary_yaml.exists():
            data = yaml.safe_load(primary_yaml.read_text(encoding="utf-8")) or {}
            return data.get("project_name", base_path.name), data.get("anime_title")
        for candidate in base_path.glob("*.yaml"):
            try:
                data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if {"project_name", "anime_title"}.intersection(data.keys()):
                return data.get("project_name", base_path.name), data.get("anime_title")
        meta_json = base_path / "config/project.json"
        if meta_json.exists():
            data = json.loads(meta_json.read_text(encoding="utf-8"))
            return data.get("name", base_path.name), data.get("anime_title")
        return base_path.name, None

    def apply_theme(self, theme: str):
        style = ttk.Style()
        style.theme_use("clam")
        if theme == "dark":
            self.state.theme = "dark"
            bg = "#0f0f0f"
            fg = "#f2f2f2"
            accent = "#1f1f1f"
            active = "#2a2a2a"
            entry_bg = "#1c1c1c"
            self._apply_window_background(self.root, bg)
            style.configure("Bg.TFrame", background=bg, foreground=fg)
            style.configure("TLabel", background=bg, foreground=fg)
            style.configure("Card.TLabelframe", background=bg, foreground=fg, bordercolor=accent)
            style.configure("Card.TLabelframe.Label", background=bg, foreground=fg)
            style.configure("Card.TCheckbutton", background=bg, foreground=fg)
            style.configure("Card.TRadiobutton", background=bg, foreground=fg)
            style.configure("TButton", background=accent, foreground=fg, bordercolor=accent, focusthickness=0)
            style.map("TButton", background=[("active", active)], foreground=[("active", fg)])
            style.configure("Accent.TButton", background=accent, foreground=fg, bordercolor=accent, focusthickness=0)
            style.map("Accent.TButton", background=[("active", active)], foreground=[("active", fg)])
            style.configure("TEntry", fieldbackground=entry_bg, background=entry_bg, foreground=fg)
            style.configure("TCombobox", fieldbackground=entry_bg, background=entry_bg, foreground=fg)
            style.map("TCombobox", fieldbackground=[("readonly", entry_bg)], background=[("active", active)])
            self.log_widget.configure(bg="#1a1a1a", fg=fg, insertbackground=fg, highlightbackground=bg, highlightcolor=bg)
        else:
            self.state.theme = "light"
            bg = "white"
            fg = "black"
            accent = "#e0e0e0"
            self._apply_window_background(self.root, bg)
            style.configure("Bg.TFrame", background=bg, foreground=fg)
            style.configure("TLabel", background=bg, foreground=fg)
            style.configure("Card.TLabelframe", background=bg, foreground=fg, bordercolor=accent)
            style.configure("Card.TLabelframe.Label", background=bg, foreground=fg)
            style.configure("Card.TCheckbutton", background=bg, foreground=fg)
            style.configure("Card.TRadiobutton", background=bg, foreground=fg)
            style.configure("TButton", background="#f4f4f4", foreground=fg, bordercolor=accent, focusthickness=0)
            style.map("TButton", background=[("active", "#e5e5e5")], foreground=[("active", fg)])
            style.configure("Accent.TButton", background="#f0f0f0", foreground=fg, bordercolor=accent, focusthickness=0)
            style.map("Accent.TButton", background=[("active", "#e5e5e5")], foreground=[("active", fg)])
            style.configure("TEntry", fieldbackground="white", background="white", foreground=fg)
            style.configure("TCombobox", fieldbackground="white", background="white", foreground=fg)
            self.log_widget.configure(bg="white", fg=fg, insertbackground=fg, highlightbackground=bg, highlightcolor=bg)
        self._style_menus(bg, fg)
        for dialog in list(self.dialogs):
            if dialog.winfo_exists():
                self._apply_window_background(dialog, bg if self.state.theme == "dark" else "white")
        self._save_state()

    def _apply_window_background(self, window: tk.Tk | tk.Toplevel, color: str | None = None):
        bg_color = color or ("#0f0f0f" if self.state.theme == "dark" else "white")
        window.configure(bg=bg_color)

    def _style_menus(self, bg: str, fg: str):
        menu_opts = {
            "background": bg,
            "foreground": fg,
            "activebackground": bg,
            "activeforeground": fg,
            "relief": "flat",
            "borderwidth": 0,
        }
        for menu in [self.menubar, self.project_menu, self.file_menu, self.view_menu, self.options_menu]:
            if menu:
                menu.configure(**menu_opts)

    def _ensure_path_vars(self):
        if self.path_vars:
            return
        for key in self.path_manager.paths_yaml.keys():
            self.path_vars[key] = tk.StringVar(value=self.path_manager.as_display_value(key))

    def _refresh_path_vars(self, reset: bool = False):
        self.base_var.set(str(self.path_manager.base_dir))
        if reset:
            self.path_vars.clear()
            self.resolved_labels.clear()
        if not self.path_vars:
            self._ensure_path_vars()
        for key, var in self.path_vars.items():
            var.set(self.path_manager.as_display_value(key))
            self.path_last_values[key] = var.get()
        self._refresh_available_stems()

    def _close_dialog(self, dialog: tk.Toplevel):
        if dialog in self.dialogs:
            self.dialogs.remove(dialog)
        dialog.destroy()

    def _choose_base_dir(self):
        chosen = filedialog.askdirectory(initialdir=str(self.path_manager.base_dir))
        if chosen:
            self.base_var.set(chosen)
            self.path_manager.base_dir = Path(chosen)
            self.state.path_state["base_dir"] = chosen
            self._refresh_all_resolved()
            self._update_log_destination()
            self._save_state()

    def _on_base_change(self):
        try:
            new_base = Path(self.base_var.get())
        except OSError:
            return
        if new_base == self.path_manager.base_dir:
            return
        self.path_manager.base_dir = new_base
        self.state.path_state["base_dir"] = str(new_base)
        self._refresh_all_resolved()
        self._update_log_destination()
        self._save_state()

    def _save_paths_yaml(self):
        self._ensure_path_vars()
        for key, var in self.path_vars.items():
            self.path_manager.update_value(key, var.get())
        self.path_manager.base_dir = Path(self.base_var.get())
        self.state.path_state = self.path_manager.to_state()
        self.path_manager.save_to_yaml()
        self._update_log_destination()
        self._save_state()
        messagebox.showinfo("Chemins", "paths.yaml mis à jour et enregistré.")

    def _choose_path_dir(self, key: str):
        initial_dir = self.path_manager.get_path(key)
        selected = filedialog.askdirectory(initialdir=str(initial_dir))
        if selected:
            old_value = self.path_vars[key].get()
            self.path_vars[key].set(selected)
            self._on_path_change(key, previous=old_value)

    def _relative_suffix(self, value: str, prefix: str) -> Path | None:
        value_parts = Path(value).parts
        prefix_parts = Path(prefix).parts
        if prefix_parts and len(value_parts) >= len(prefix_parts) and list(value_parts[: len(prefix_parts)]) == list(prefix_parts):
            return Path(*value_parts[len(prefix_parts) :])
        return None

    def _on_path_change(self, key: str, previous: str | None = None):
        current_value = self.path_vars[key].get()
        old_value = previous if previous is not None else self.path_last_values.get(key, current_value)
        if current_value == old_value:
            return
        self.path_manager.update_value(key, current_value)
        self.path_last_values[key] = current_value
        if key in self.path_manager.children_map:
            for child in self.path_manager.children_map[key]:
                child_var = self.path_vars[child]
                child_value = child_var.get()
                if Path(child_value).is_absolute():
                    continue
                suffix = self._relative_suffix(child_value, old_value)
                if suffix is None:
                    continue
                new_child_value = Path(current_value) / suffix
                child_var.set(str(new_child_value))
                self.path_manager.update_value(child, child_var.get())
                self.path_last_values[child] = child_var.get()
        self._refresh_all_resolved()
        self.state.path_state = self.path_manager.to_state()
        self._save_state()

    def _refresh_all_resolved(self):
        for key in self.resolved_labels:
            self._refresh_resolved_for_key(key)

    def _refresh_resolved_for_key(self, key: str):
        if key not in self.resolved_labels:
            return
        resolved = self.path_manager.get_path(key)
        self.resolved_labels[key].configure(text=str(resolved))


def main():
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
