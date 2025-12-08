"""
Interface GUI pour orchestrer les étapes du pipeline.

Fonctionnalités principales :
- Sélectionner les étapes à exécuter et l'ordre par défaut 01→09.
- Choisir le niveau de pause (après chaque fichier, répertoire ou étape).
- Modifier les fichiers de configuration (chemins de données) directement depuis l'interface.
- Enregistrer / charger l'état complet : thème, chemins, options, étape + fichier en cours.
- Relancer automatiquement les scripts concernés avec filtrage ``--stem`` pour reprendre sur un fichier précis.

Le GUI est basé sur Tkinter afin d'éviter de nouvelles dépendances.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
    WorkflowStep("01", "Extraction audio", "01_extract_audio.py", "episodes_raw_dir", "*.mkv", "ffmpeg → wav"),
    WorkflowStep("02", "Diarisation", "02_diarize.py", "audio_raw_dir", "*_mono16k.wav", "pyannote 3.1"),
    WorkflowStep("03", "Transcription Whisper", "03_whisper_transcribe.py", "audio_raw_dir", "*_mono16k.wav", "Whisper large-v3"),
    WorkflowStep("04", "Traduction NLLB", "04_translate_nllb.py", "whisper_json_dir", "*.json", "NLLB 600M"),
    WorkflowStep("05", "Banque de voix", "05_build_speaker_bank.py", None, None, "Initialisation/embeddings"),
    WorkflowStep("06", "Attribution personnages", "06_assign_characters.py", "whisper_json_fr_dir", "*_fr.json", "Matching embeddings"),
    WorkflowStep("07", "Synthèse XTTS", "07_synthesize_xtts.py", "segments_dir", "*_segments.json", "XTTS-v2"),
    WorkflowStep("08", "Mix audio", "08_mix_audio.py", "dub_audio_dir", "*_fr_voices.wav", "amix ffmpeg"),
    WorkflowStep("09", "Remux vidéo", "09_remux.py", "dub_audio_dir", "*_fr_full.wav", "Remux MKV"),
]


class PathManager:
    """Gère les chemins (base + overrides) et la sauvegarde YAML."""

    def __init__(self, base_dir: Path, overrides: Dict[str, str] | None = None):
        self.base_dir = base_dir
        self.overrides = overrides or {}
        self.paths_yaml = load_paths_config()

    def as_display_value(self, key: str) -> str:
        if key in self.overrides:
            return self.overrides[key]
        return self.paths_yaml.get(key, "")

    def get_path(self, key: str) -> Path:
        raw_value = self.as_display_value(key)
        path = Path(raw_value)
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    def update_value(self, key: str, value: str) -> None:
        self.overrides[key] = value

    def save_to_yaml(self) -> None:
        updated = dict(self.paths_yaml)
        updated.update(self.overrides)
        for k, v in updated.items():
            if Path(v).is_absolute():
                # Stocke les chemins absolus tels quels ; les autres resteront relatifs à base_dir
                updated[k] = str(v)
        (PROJECT_ROOT / "config/paths.yaml").write_text(
            yaml.safe_dump(updated, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def to_state(self) -> dict:
        return {"base_dir": str(self.base_dir), "overrides": self.overrides}


class WorkflowState:
    def __init__(self):
        self.theme = "dark"
        self.pause_mode = "step"  # values: file, directory, step
        self.selected_steps = [s.step_id for s in STEPS]
        self.current_step = None
        self.current_index = 0
        self.path_state = {"base_dir": str(PROJECT_ROOT), "overrides": {}}

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

    def save(self, path: Path = STATE_PATH) -> None:
        payload = {
            "theme": self.theme,
            "pause_mode": self.pause_mode,
            "selected_steps": self.selected_steps,
            "current_step": self.current_step,
            "current_index": self.current_index,
            "paths": self.path_state,
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

            units = step.list_units(self.path_manager)
            start_index = self.state.current_index if self.state.current_step == step.step_id else 0
            for idx, unit in enumerate(units):
                if idx < start_index:
                    continue
                if self.stop_event.is_set():
                    self.log("Arrêt demandé, sauvegarde de l'état…")
                    self.state.current_step = step.step_id
                    self.state.current_index = idx
                    self.state.save()
                    return

                self.state.current_step = step.step_id
                self.state.current_index = idx
                self.state.save()

                self.log(f"→ {step.label} : {unit}")
                self._run_script(step, unit)

                if self.pause_event.is_set():
                    self.log("Mise en pause demandée.")
                    self.state.save()
                    return

                if self.state.pause_mode == "file" and unit != "_all_":
                    self.log("Pause après fichier (option active).")
                    self.pause_event.set()
                    self.state.save()
                    return

            if self.state.pause_mode == "directory":
                self.log("Pause après répertoire (option active).")
                self.pause_event.set()
                self.state.save()
                return

            if self.state.pause_mode == "step":
                self.log("Pause après étape (option active).")
                self.pause_event.set()
                self.state.save()
                return

        self.log("Pipeline terminé.")
        self.state.current_step = None
        self.state.current_index = 0
        self.state.save()

    def _run_script(self, step: WorkflowStep, unit: str):
        script_path = PROJECT_ROOT / "scripts" / step.script
        cmd = ["python", str(script_path)]
        if unit != "_all_":
            cmd.extend(["--stem", unit])
        try:
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if completed.stdout:
                self.log(completed.stdout.strip())
            if completed.stderr:
                self.log(completed.stderr.strip())
        except subprocess.CalledProcessError as exc:
            self.log(f"Erreur lors de {step.label} ({unit}) : {exc}")
            self.pause_event.set()
            self.state.save()

    def stop(self):
        self.stop_event.set()

    def pause(self):
        self.pause_event.set()


class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Anime Dub – Orchestrateur GUI")
        self.state = WorkflowState()
        self.state.load()

        base_dir = Path(self.state.path_state.get("base_dir", PROJECT_ROOT))
        overrides = self.state.path_state.get("overrides", {})
        self.path_manager = PathManager(base_dir=base_dir, overrides=overrides)

        self.runner = WorkflowRunner(self.log, self.state, self.path_manager)
        self.step_vars: Dict[str, tk.BooleanVar] = {}
        self.path_vars: Dict[str, tk.StringVar] = {}
        self.base_var = tk.StringVar(value=str(self.path_manager.base_dir))
        self.dialogs: set[tk.Toplevel] = set()

        self._build_menu()
        self._build_layout()
        self.apply_theme(self.state.theme)

    def _build_menu(self):
        menu = tk.Menu(self.root)

        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Enregistrer l'état", command=self.save_state)
        file_menu.add_command(label="Charger un état", command=self.load_state)
        file_menu.add_command(label="Enregistrer sous…", command=self.save_state_as)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.destroy)
        menu.add_cascade(label="Fichier", menu=file_menu)

        view_menu = tk.Menu(menu, tearoff=0)
        view_menu.add_command(label="Mode Nuit", command=lambda: self.apply_theme("dark"))
        view_menu.add_command(label="Mode Jour", command=lambda: self.apply_theme("light"))
        menu.add_cascade(label="Affichage", menu=view_menu)

        options_menu = tk.Menu(menu, tearoff=0)
        options_menu.add_command(label="Configurer les chemins…", command=self.open_paths_window)
        options_menu.add_separator()
        pause_menu = tk.Menu(options_menu, tearoff=0)
        pause_menu.add_command(label="Pause par fichier", command=lambda: self._set_pause("file"))
        pause_menu.add_command(label="Pause par répertoire", command=lambda: self._set_pause("directory"))
        pause_menu.add_command(label="Pause par étape", command=lambda: self._set_pause("step"))
        options_menu.add_cascade(label="Mode de pause", menu=pause_menu)
        menu.add_cascade(label="Options", menu=options_menu)

        self.root.config(menu=menu)

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        # Section étapes
        steps_frame = ttk.LabelFrame(container, text="Étapes du workflow")
        steps_frame.pack(fill=tk.X, expand=False, pady=5)

        for idx, step in enumerate(STEPS):
            var = tk.BooleanVar(value=step.step_id in self.state.selected_steps)
            self.step_vars[step.step_id] = var
            ttk.Checkbutton(steps_frame, text=f"{step.step_id} – {step.label} ({step.description})", variable=var).grid(row=idx, column=0, sticky="w")

        # Section commandes
        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=5)

        ttk.Button(controls, text="Démarrer / Reprendre", command=self.start_workflow).pack(side=tk.LEFT)
        ttk.Button(controls, text="Pause", command=self.runner.pause).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Arrêter", command=self.runner.stop).pack(side=tk.LEFT)

        ttk.Label(controls, text="Mode de pause :").pack(side=tk.LEFT, padx=(20, 4))
        self.pause_var = tk.StringVar(value=self.state.pause_mode)
        pause_combo = ttk.Combobox(controls, textvariable=self.pause_var, values=["file", "directory", "step"], width=12)
        pause_combo.pack(side=tk.LEFT)

        # Log
        log_frame = ttk.LabelFrame(container, text="Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_widget = tk.Text(log_frame, height=15, wrap="word")
        self.log_widget.pack(fill=tk.BOTH, expand=True)

    def log(self, message: str):
        self.log_widget.insert(tk.END, message + "\n")
        self.log_widget.see(tk.END)

    def open_paths_window(self):
        if hasattr(self, "paths_window") and self.paths_window.winfo_exists():
            self.paths_window.focus_set()
            return

        self.paths_window = tk.Toplevel(self.root)
        self.paths_window.title("Chemins et répertoires")
        self.dialogs.add(self.paths_window)
        self.paths_window.protocol("WM_DELETE_WINDOW", lambda: self._close_dialog(self.paths_window))
        container = ttk.Frame(self.paths_window, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Répertoire de base").grid(row=0, column=0, sticky="w")
        ttk.Entry(container, textvariable=self.base_var, width=80).grid(row=0, column=1, sticky="ew")
        ttk.Button(container, text="Parcourir", command=self._choose_base_dir).grid(row=0, column=2, padx=4)
        container.columnconfigure(1, weight=1)

        self._ensure_path_vars()
        row = 1
        for key, var in self.path_vars.items():
            ttk.Label(container, text=key).grid(row=row, column=0, sticky="w")
            ttk.Entry(container, textvariable=var, width=80).grid(row=row, column=1, sticky="ew")
            row += 1

        ttk.Button(container, text="Enregistrer paths.yaml", command=self._save_paths_yaml).grid(row=row, column=0, columnspan=3, pady=6, sticky="e")
        self._apply_window_background(self.paths_window)

    def _set_pause(self, mode: str):
        self.pause_var.set(mode)
        self.state.pause_mode = mode
        self.state.save()

    def start_workflow(self):
        selected = [step_id for step_id, var in self.step_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("Étapes", "Sélectionne au moins une étape à exécuter.")
            return
        self.state.selected_steps = selected
        self.state.path_state = self.path_manager.to_state()
        self.state.pause_mode = self.pause_var.get()
        self.state.save()
        self.runner.start(selected, self.pause_var.get())

    def save_state(self):
        self.state.path_state = self.path_manager.to_state()
        self.state.save()
        messagebox.showinfo("État", f"Sauvegarde enregistrée dans {STATE_PATH}")

    def save_state_as(self):
        dest = filedialog.asksaveasfilename(defaultextension=".json", initialdir=str(PROJECT_ROOT / "config"))
        if not dest:
            return
        self.state.path_state = self.path_manager.to_state()
        self.state.save(Path(dest))

    def load_state(self):
        src = filedialog.askopenfilename(defaultextension=".json", initialdir=str(PROJECT_ROOT / "config"))
        if not src:
            return
        self.state.load(Path(src))
        self.pause_var.set(self.state.pause_mode)
        self.apply_theme(self.state.theme)
        self.path_manager = PathManager(base_dir=Path(self.state.path_state.get("base_dir", PROJECT_ROOT)), overrides=self.state.path_state.get("overrides", {}))
        self._refresh_path_vars()
        for step in STEPS:
            self.step_vars[step.step_id].set(step.step_id in self.state.selected_steps)
        self.log("État chargé.")

    def apply_theme(self, theme: str):
        style = ttk.Style()
        if theme == "dark":
            self.state.theme = "dark"
            bg = "#0f0f0f"
            fg = "#f2f2f2"
            accent = "#1c1c1c"
            entry_bg = "#1f1f1f"
            self._apply_window_background(self.root, bg)
            style.configure("TFrame", background=bg, foreground=fg)
            style.configure("TLabel", background=bg, foreground=fg)
            style.configure("TButton", background=accent, foreground=fg)
            style.configure("TLabelFrame", background=bg, foreground=fg)
            style.configure("TCheckbutton", background=bg, foreground=fg)
            style.configure("TEntry", fieldbackground=entry_bg, background=entry_bg, foreground=fg)
            style.configure("TCombobox", fieldbackground=entry_bg, background=entry_bg, foreground=fg)
            self.log_widget.configure(bg="#1a1a1a", fg=fg, insertbackground=fg)
        else:
            self.state.theme = "light"
            bg = "white"
            fg = "black"
            self._apply_window_background(self.root, bg)
            style.configure("TFrame", background=bg, foreground=fg)
            style.configure("TLabel", background=bg, foreground=fg)
            style.configure("TButton", background="#f0f0f0", foreground=fg)
            style.configure("TLabelFrame", background=bg, foreground=fg)
            style.configure("TCheckbutton", background=bg, foreground=fg)
            style.configure("TEntry", fieldbackground="white", background="white", foreground=fg)
            style.configure("TCombobox", fieldbackground="white", background="white", foreground=fg)
            self.log_widget.configure(bg="white", fg=fg, insertbackground=fg)
        for dialog in list(self.dialogs):
            if dialog.winfo_exists():
                self._apply_window_background(dialog, bg if self.state.theme == "dark" else "white")
        self.state.save()

    def _apply_window_background(self, window: tk.Tk | tk.Toplevel, color: str | None = None):
        bg_color = color or ("#0f0f0f" if self.state.theme == "dark" else "white")
        window.configure(bg=bg_color)

    def _ensure_path_vars(self):
        if self.path_vars:
            return
        for key in load_paths_config().keys():
            self.path_vars[key] = tk.StringVar(value=self.path_manager.as_display_value(key))

    def _refresh_path_vars(self):
        self.base_var.set(str(self.path_manager.base_dir))
        if not self.path_vars:
            self._ensure_path_vars()
        for key, var in self.path_vars.items():
            var.set(self.path_manager.as_display_value(key))

    def _close_dialog(self, dialog: tk.Toplevel):
        if dialog in self.dialogs:
            self.dialogs.remove(dialog)
        dialog.destroy()

    def _choose_base_dir(self):
        chosen = filedialog.askdirectory()
        if chosen:
            self.base_var.set(chosen)
            self.path_manager.base_dir = Path(chosen)
            self.state.path_state["base_dir"] = chosen
            self.state.save()

    def _save_paths_yaml(self):
        self._ensure_path_vars()
        for key, var in self.path_vars.items():
            self.path_manager.update_value(key, var.get())
        self.path_manager.base_dir = Path(self.base_var.get())
        self.state.path_state = self.path_manager.to_state()
        self.path_manager.save_to_yaml()
        self.state.save()
        messagebox.showinfo("Chemins", "paths.yaml mis à jour et enregistré.")


def main():
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
