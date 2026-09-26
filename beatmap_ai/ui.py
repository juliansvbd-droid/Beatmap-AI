"""A beginner-friendly Windows desktop UI for BeatMap-AI."""

from __future__ import annotations

import os
import math
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "beatmap_ai" / "models" / "rhythm.pt"
SEQUENCE_PATH = PROJECT_ROOT / "beatmap_ai" / "models" / "sequence.pt"
PLACEMENT_PATH = PROJECT_ROOT / "beatmap_ai" / "models" / "placement.pt"
DATA_PATH = PROJECT_ROOT / "data"
# (label, --style name); the first entry means "Auto" (no style).
STYLE_CHOICES = (
    ("Auto (KI entscheidet)", None),
    ("Jump – große Sprünge", "jump"),
    ("Stream – schnelle Notenketten", "stream"),
    ("Tech – vielseitige Rhythmen", "tech"),
    ("Flow – ruhig, viele Slider", "flow"),
    ("Circles – kaum Slider", "circles"),
    ("Cross-Screen – weite Sprünge", "cross-screen"),
    ("Doubles – Doppel-Taps", "doubles"),
    ("Alt – Wechsel-Tapping", "alt"),
    ("Geometric – saubere Formen", "geometric"),
    ("Simple – ruhig und klar", "simple"),
)
# Training folders preselected in the UI (several are separated by ";").
DATA_PATHS = (DATA_PATH, PROJECT_ROOT / "best_maps")
CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"

BG = "#F3F5FA"
SURFACE = "#FFFFFF"
TEXT = "#192333"
MUTED = "#637187"
ACCENT = "#5B55E7"
ACCENT_HOVER = "#4B45CF"
LINE = "#E1E6EF"
FIELD = "#F8FAFD"
GREEN = "#117A5A"


def _console_python() -> str:
    """Use python.exe for captured CLI output even when the UI was opened with pythonw."""
    executable = Path(sys.executable)
    if executable.name.lower().startswith("pythonw"):
        candidate = executable.with_name(executable.name.replace("pythonw", "python", 1))
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _detect_backend() -> tuple[str, str]:
    """Return a usable training backend and a short user-facing description."""
    try:
        import torch

        if torch.cuda.is_available():
            api = "ROCm" if torch.version.hip else "CUDA"
            return "cuda", f"GPU bereit: {torch.cuda.get_device_name(0)} ({api})"
    except Exception:
        pass

    try:
        import torch_directml

        count = torch_directml.device_count()
        if count:
            return "directml", f"GPU bereit: {torch_directml.device_name(0)} (DirectML)"
    except Exception:
        pass

    return "cpu", "Keine GPU erkannt – Training läuft auf der CPU."


class BeatmapApp(tk.Tk):
    def __init__(self, initial_audio: str | Path | None = None) -> None:
        super().__init__()
        self.title("BeatMap AI")
        self.geometry("1040x870")
        self.minsize(900, 740)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None
        self.active_task: str | None = None
        self.cancel_requested = False
        self.last_generated: Path | None = None
        self.last_checkpoint: Path | None = None
        self.last_learning_report: Path | None = None
        self.detected_backend = "cpu"

        self._configure_styles()
        self._build_layout()
        if initial_audio:
            audio_p = Path(initial_audio).resolve()
            if audio_p.is_file():
                self.audio_var.set(str(audio_p))
                self.output_map_var.set(str(audio_p.with_suffix(".osz")))
        self._load_recent_learning_report()
        self.after(100, self._poll_events)
        threading.Thread(target=self._refresh_hardware_status, daemon=True).start()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Card.TFrame", background=SURFACE)
        style.configure("Card.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("CardMuted.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 23, "bold"))
        style.configure("Heading.TLabel", background=SURFACE, foreground=TEXT,
                        font=("Segoe UI", 12, "bold"))
        style.configure("TEntry", padding=(8, 7), fieldbackground=FIELD, foreground=TEXT)
        style.configure("TCombobox", padding=(7, 6), fieldbackground=FIELD, foreground=TEXT)
        style.configure("TCheckbutton", background=SURFACE, foreground=TEXT,
                        font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", SURFACE)])
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 8, 0))
        style.configure("TNotebook.Tab", padding=(16, 10), background="#E9ECF4",
                        foreground=MUTED, font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", SURFACE)],
                  foreground=[("selected", ACCENT)])
        style.configure("Primary.TButton", padding=(16, 10), background=ACCENT,
                        foreground="#FFFFFF", borderwidth=0, font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#B8B9CB")])
        style.configure("Secondary.TButton", padding=(11, 7), background="#EEF0F7",
                        foreground=TEXT, borderwidth=0, font=("Segoe UI", 9, "bold"))
        style.map("Secondary.TButton", background=[("active", "#E2E5F0")])
        style.configure("Success.TButton", padding=(12, 8), background=GREEN,
                        foreground="#FFFFFF", borderwidth=0, font=("Segoe UI", 9, "bold"))
        style.configure("Horizontal.TProgressbar", troughcolor="#E7EAF2",
                        background=ACCENT, bordercolor="#E7EAF2", lightcolor=ACCENT,
                        darkcolor=ACCENT)

    def _build_layout(self) -> None:
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 12))
        left = tk.Frame(header, bg=BG)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="BeatMap AI", bg=BG, fg=TEXT,
                 font=("Segoe UI", 23, "bold")).pack(anchor="w")
        tk.Label(left, text="Erstelle osu!‑Beatmaps aus deiner Musik – Schritt für Schritt.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 0))
        ttk.Button(header, text="Hilfe öffnen", style="Secondary.TButton",
                   command=self._open_help).pack(side="right", anchor="n", pady=7)

        # Footer and log are packed from the bottom first, so they stay visible when the
        # window is small; the pages above scroll instead.
        log_frame = tk.Frame(self, bg=SURFACE, highlightbackground=LINE, highlightthickness=1)
        log_frame.pack(side="bottom", fill="x", padx=24, pady=(4, 16))
        footer = ttk.Frame(self, padding=(24, 12, 24, 6))
        footer.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Bereit.")
        self.status_label = ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(footer, length=190, mode="determinate", maximum=100)
        self.progress.pack(side="right")

        log_header = tk.Frame(log_frame, bg=SURFACE)
        log_header.pack(fill="x", padx=12, pady=(9, 5))
        tk.Label(log_header, text="Aktivität", bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        self.cancel_button = ttk.Button(log_header, text="Aufgabe abbrechen",
                                        style="Secondary.TButton", command=self._cancel_task,
                                        state="disabled")
        self.cancel_button.pack(side="right")
        text_frame = tk.Frame(log_frame, bg=SURFACE)
        text_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.log_text = tk.Text(text_frame, height=7, wrap="word", state="disabled",
                                bg=FIELD, fg=TEXT, insertbackground=TEXT,
                                relief="flat", padx=10, pady=8,
                                font=("Consolas", 9), borderwidth=0)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._append_log("Willkommen! Wähle oben eine Aufgabe aus. Die Verarbeitung bleibt lokal auf diesem PC.")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20)
        self._scroll_canvases: list[tk.Canvas] = []
        generate_tab, self.generate_page = self._scrollable(self.notebook)
        train_tab, self.train_page = self._scrollable(self.notebook)
        self.learning_page = ttk.Frame(self.notebook, padding=(16, 14))
        self.notebook.add(generate_tab, text="Beatmap erstellen")
        self.notebook.add(train_tab, text="KI trainieren")
        self.notebook.add(self.learning_page, text="Was die KI lernt")
        self._build_generate_page()
        self._build_train_page()
        self._build_learning_page()

        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _scrollable(self, parent: ttk.Notebook) -> tuple[ttk.Frame, ttk.Frame]:
        """A notebook tab whose content scrolls vertically. Returns (tab, content frame)."""
        tab = ttk.Frame(parent)
        canvas = tk.Canvas(tab, bg=BG, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, padding=(16, 14))
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._scroll_canvases.append(canvas)
        return tab, content

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Scroll the page under the mouse (the activity log keeps its own scrolling)."""
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return
        for canvas in self._scroll_canvases:
            if str(widget).startswith(str(canvas)):
                top, bottom = canvas.yview()
                if top > 0.0 or bottom < 1.0:
                    canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")
                return

    @staticmethod
    def _card(parent: ttk.Frame, title: str, note: str | None = None) -> tk.Frame:
        card = tk.Frame(parent, bg=SURFACE, highlightbackground=LINE, highlightthickness=1)
        card.pack(fill="x", pady=(0, 11))
        inner = tk.Frame(card, bg=SURFACE)
        inner.pack(fill="x", padx=15, pady=12)
        tk.Label(inner, text=title, bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        if note:
            tk.Label(inner, text=note, bg=SURFACE, fg=MUTED,
                     font=("Segoe UI", 9), justify="left", wraplength=880).pack(anchor="w", pady=(3, 8))
        return inner

    def _path_row(self, parent: tk.Frame, variable: tk.StringVar, button_text: str,
                  command, *, entry_width: int = 78) -> ttk.Entry:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(2, 0))
        entry = ttk.Entry(row, textvariable=variable, width=entry_width)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text=button_text, style="Secondary.TButton", command=command).pack(side="right")
        return entry

    def _build_generate_page(self) -> None:
        self.audio_var = tk.StringVar()
        self.output_map_var = tk.StringVar()
        self.artist_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.bpm_var = tk.StringVar()
        self.offset_var = tk.StringVar()
        self.use_model_var = tk.BooleanVar(value=True)
        self.difficulties = {
            "easy": tk.BooleanVar(value=False),
            "normal": tk.BooleanVar(value=True),
            "hard": tk.BooleanVar(value=True),
            "insane": tk.BooleanVar(value=True),
            "expert": tk.BooleanVar(value=False),
        }
        self.stars_var = tk.StringVar()
        self.style_var = tk.StringVar(value=STYLE_CHOICES[0][0])
        self.style_strength_var = tk.DoubleVar(value=70)
        self.style_strength_label = tk.StringVar(value="70 %")
        self.variety_var = tk.DoubleVar(value=50)
        self.variety_label = tk.StringVar(value="50 %")
        self.critic_var = tk.BooleanVar(value=True)

        card = self._card(self.generate_page, "1. Song auswählen",
                          "Unterstützt MP3 und OGG. Der Künstler und Titel können aus „Künstler - Titel.mp3“ übernommen werden.")
        self._path_row(card, self.audio_var, "Datei auswählen…", self._choose_audio)

        card = self._card(self.generate_page, "2. Schwierigkeiten",
                          "Wähle eine oder mehrere Versionen für dein Beatmap-Set.")
        chips = ttk.Frame(card, style="Card.TFrame")
        chips.pack(anchor="w")
        labels = {"easy": "Easy", "normal": "Normal", "hard": "Hard",
                  "insane": "Insane", "expert": "Expert"}
        for key, label in labels.items():
            ttk.Checkbutton(chips, text=label, variable=self.difficulties[key]).pack(side="left", padx=(0, 16))
        stars_row = ttk.Frame(card, style="Card.TFrame")
        stars_row.pack(fill="x", pady=(8, 0))
        self._labeled_entry(stars_row, "Sternzahlen (optional, z. B. 3.8; 4.5; 6)", self.stars_var, 0, 0, width=30)
        tk.Label(card, text="Jede Sternzahl wird eine eigene Difficulty. BeatMap AI rechnet mit der "
                            "osu!-Sternformel nach und passt die Map an, bis sie stimmt.",
                 bg=SURFACE, fg=MUTED, font=("Segoe UI", 9), wraplength=720, justify="left").pack(anchor="w", pady=(4, 0))

        card = self._card(self.generate_page, "Stil (optional)",
                          "Auto lässt die KI selbst entscheiden, was zur Musik passt. "
                          "Ein Stil gibt der Map zusätzlich einen Charakter.")
        style_row = ttk.Frame(card, style="Card.TFrame")
        style_row.pack(fill="x")
        ttk.Combobox(style_row, textvariable=self.style_var, state="readonly", width=28,
                     values=[label for label, _ in STYLE_CHOICES]).pack(side="left")
        tk.Label(style_row, text="Stärke", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(16, 6))
        ttk.Scale(style_row, from_=0, to=100, variable=self.style_strength_var, orient="horizontal",
                  length=180, command=lambda v: self.style_strength_label.set(f"{float(v):.0f} %")).pack(
            side="left")
        tk.Label(style_row, textvariable=self.style_strength_label, bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9), width=5).pack(side="left", padx=(6, 0))
        variety_row = ttk.Frame(card, style="Card.TFrame")
        variety_row.pack(fill="x", pady=(10, 0))
        tk.Label(variety_row, text="Rhythmus-Vielfalt:  nah am Beat", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        ttk.Scale(variety_row, from_=0, to=100, variable=self.variety_var, orient="horizontal",
                  length=180, command=lambda v: self.variety_label.set(f"{float(v):.0f} %")).pack(
            side="left", padx=(6, 6))
        tk.Label(variety_row, text="abwechslungsreich", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(variety_row, textvariable=self.variety_label, bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9), width=5).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(card, text="Bewerter-KI (wählt die menschlichste von 4 Platzierungen)",
                        variable=self.critic_var).pack(anchor="w", pady=(10, 0))

        card = self._card(self.generate_page, "3. Ausgabe und Extras",
                          "Leere Metadatenfelder werden automatisch aus dem Dateinamen ausgefüllt.")
        metadata = ttk.Frame(card, style="Card.TFrame")
        metadata.pack(fill="x", pady=(0, 8))
        self._labeled_entry(metadata, "Künstler (optional)", self.artist_var, 0, 0)
        self._labeled_entry(metadata, "Songtitel (optional)", self.title_var, 0, 1)
        ttk.Checkbutton(card, text="KI verwenden (empfohlen: Sequence v2 + Conformer v4)",
                        variable=self.use_model_var).pack(anchor="w", pady=(0, 2))
        tk.Label(card, text="Aktiv: Sequence v2 (Rhythmus + Platzierung) mit GPU-Beschleunigung",
                 bg=SURFACE, fg=GREEN, font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 8))

        tk.Label(card, text="Beatmap-Datei (.osz)", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self._path_row(card, self.output_map_var, "Speicherort…", self._choose_map_output)

        self.advanced_generate_open = False
        self.advanced_generate = ttk.Frame(card, style="Card.TFrame")
        ttk.Button(card, text="Timing-Einstellungen anzeigen", style="Secondary.TButton",
                   command=self._toggle_generate_advanced).pack(anchor="w", pady=(9, 0))

        timing = ttk.Frame(self.advanced_generate, style="Card.TFrame")
        timing.pack(fill="x", pady=(9, 0))
        self._labeled_entry(timing, "BPM überschreiben", self.bpm_var, 0, 0, width=18)
        self._labeled_entry(timing, "Offset in ms überschreiben", self.offset_var, 0, 1, width=18)
        tk.Label(self.advanced_generate,
                 text="Leer lassen, damit BeatMap AI Tempo und Offset automatisch erkennt.",
                 bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(5, 0))

        actions = ttk.Frame(self.generate_page)
        actions.pack(fill="x", pady=(2, 0))
        self.generate_button = ttk.Button(actions, text="Beatmap erstellen",
                                          style="Primary.TButton", command=self._start_generate)
        self.generate_button.pack(side="left")
        self.open_osu_button = ttk.Button(actions, text="In osu! importieren",
                                          style="Success.TButton", command=self._open_generated,
                                          state="disabled")
        self.open_osu_button.pack(side="left", padx=(8, 0))
        self.open_map_folder_button = ttk.Button(actions, text="Ordner öffnen",
                                                 style="Secondary.TButton", command=self._open_generated_folder,
                                                 state="disabled")
        self.open_map_folder_button.pack(side="left", padx=(8, 0))

    def _build_train_page(self) -> None:
        self.data_var = tk.StringVar(value=";".join(str(p) for p in DATA_PATHS if p.exists()))
        self.init_model_var = tk.StringVar(value=str(MODEL_PATH))
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        self.model_output_var = tk.StringVar(value=str(CHECKPOINTS_PATH / f"rhythm-finetuned-{stamp}.pt"))
        self.continue_model_var = tk.BooleanVar(value=True)
        self.training_mode_var = tk.StringVar(value="Ausgewogen (empfohlen)")
        self.device_choice_var = tk.StringVar(value="Automatisch")
        self.epochs_var = tk.StringVar(value="12")
        self.steps_var = tk.StringVar(value="50")
        self.batch_var = tk.StringVar(value="4")
        self.lr_var = tk.StringVar(value="0.0001")
        self.workers_var = tk.StringVar(value="4")

        card = self._card(self.train_page, "1. Beatmap-Sammlung auswählen",
                          "Wähle deinen osu! Songs-Ordner oder einen Ordner mit .osz-Dateien. "
                          "Mehrere Ordner trennst du mit ; – doppelte Maps werden nur einmal genutzt.")
        self._path_row(card, self.data_var, "Ordner auswählen…", self._choose_data_folder)

        card = self._card(self.train_page, "2. Startmodell und Ergebnis",
                          "Beim Weitertrainieren lernt das Modell ausgehend von den vorhandenen Gewichten.")
        ttk.Checkbutton(card, text="Mit aktuellem Projektmodell weitertrainieren",
                        variable=self.continue_model_var,
                        command=self._update_init_model_state).pack(anchor="w", pady=(0, 7))
        self.init_model_entry = self._path_row(card, self.init_model_var, "Checkpoint auswählen…",
                                               self._choose_init_model)
        tk.Label(card, text="Neues Modell speichern unter (.pt)", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(10, 0))
        self._path_row(card, self.model_output_var, "Speicherort…", self._choose_model_output)

        card = self._card(self.train_page, "3. Trainingsumfang",
                          "Die ausgewogene Einstellung ist für ein erstes Fine-Tuning gedacht. "
                          "Die Details kannst du bei Bedarf unten anpassen.")
        self.training_mode_box = ttk.Combobox(
            card, textvariable=self.training_mode_var, state="readonly", width=29,
            values=("Kurz", "Ausgewogen (empfohlen)", "Individuell"))
        self.training_mode_box.pack(anchor="w")
        self.training_mode_box.bind("<<ComboboxSelected>>", self._on_training_mode_change)
        self.hardware_var = tk.StringVar(value="GPU wird gesucht …")
        tk.Label(card, textvariable=self.hardware_var, bg=SURFACE, fg=GREEN,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(7, 0))

        self.advanced_train_open = False
        self.advanced_train = ttk.Frame(card, style="Card.TFrame")
        self._labeled_entry(self.advanced_train, "Epochen", self.epochs_var, 0, 0, width=12)
        self._labeled_entry(self.advanced_train, "Schritte je Epoche", self.steps_var, 0, 1, width=12)
        self._labeled_entry(self.advanced_train, "Batchgröße", self.batch_var, 0, 2, width=12)
        self._labeled_entry(self.advanced_train, "Lernrate", self.lr_var, 1, 0, width=12)
        self._labeled_entry(self.advanced_train, "CPU-Prozesse für Audio", self.workers_var, 1, 1, width=12)
        device_frame = ttk.Frame(self.advanced_train, style="Card.TFrame")
        device_frame.grid(row=1, column=2, sticky="nw", padx=(14, 0), pady=(9, 0))
        ttk.Label(device_frame, text="Rechengerät", style="CardMuted.TLabel").pack(anchor="w")
        ttk.Combobox(device_frame, textvariable=self.device_choice_var, state="readonly", width=24,
                     values=("Automatisch", "DirectML (AMD/Windows)", "CUDA (NVIDIA/ROCm)", "CPU"))\
            .pack(anchor="w", pady=(3, 0))
        ttk.Button(card, text="Weitere Einstellungen anzeigen", style="Secondary.TButton",
                   command=self._toggle_train_advanced).pack(anchor="w", pady=(9, 0))

        actions = ttk.Frame(self.train_page)
        actions.pack(fill="x", pady=(2, 0))
        self.train_button = ttk.Button(actions, text="Training starten",
                                       style="Primary.TButton", command=self._start_train)
        self.train_button.pack(side="left")
        self.activate_model_button = ttk.Button(actions, text="Als Standardmodell übernehmen",
                                                style="Success.TButton", command=self._activate_model,
                                                state="disabled")
        self.activate_model_button.pack(side="left", padx=(8, 0))
        self.open_checkpoint_folder_button = ttk.Button(actions, text="Checkpoint-Ordner öffnen",
                                                        style="Secondary.TButton",
                                                        command=self._open_checkpoint_folder,
                                                        state="disabled")
        self.open_checkpoint_folder_button.pack(side="left", padx=(8, 0))
        self.open_learning_button = ttk.Button(
            actions, text="Lernübersicht anzeigen", style="Secondary.TButton",
            command=self._open_learning_page, state="disabled")
        self.open_learning_button.pack(side="left", padx=(8, 0))

    def _build_learning_page(self) -> None:
        intro = self._card(
            self.learning_page,
            "Das hat die KI aus dem Training mitgenommen",
            "Der Bericht beschreibt die tatsächlich verwendeten Beatmaps und zeigt, wie gut "
            "das Modell Noten auf zurückgehaltenen Songs wiedererkennt. Er macht die "
            "Gewichte nicht wörtlich lesbar, sondern erklärt die Muster, aus denen das Modell lernt.",
        )
        actions = ttk.Frame(intro, style="Card.TFrame")
        actions.pack(fill="x", pady=(2, 0))
        ttk.Button(actions, text="Trainingsbericht öffnen…", style="Secondary.TButton",
                   command=self._choose_learning_report).pack(side="left")
        self.learning_report_path_var = tk.StringVar(value="Noch kein Trainingsbericht geladen.")
        ttk.Label(actions, textvariable=self.learning_report_path_var,
                  style="CardMuted.TLabel").pack(side="left", padx=(10, 0))

        report_card = tk.Frame(self.learning_page, bg=SURFACE,
                               highlightbackground=LINE, highlightthickness=1)
        report_card.pack(fill="both", expand=True)
        text_frame = tk.Frame(report_card, bg=SURFACE)
        text_frame.pack(fill="both", expand=True, padx=12, pady=12)
        self.learning_text = tk.Text(
            text_frame, wrap="word", state="disabled", bg=FIELD, fg=TEXT,
            insertbackground=TEXT, relief="flat", padx=14, pady=12,
            font=("Segoe UI", 10), borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.learning_text.yview)
        self.learning_text.configure(yscrollcommand=scrollbar.set)
        self.learning_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.learning_text.tag_configure("title", font=("Segoe UI", 15, "bold"), spacing3=10)
        self.learning_text.tag_configure("heading", font=("Segoe UI", 11, "bold"), spacing1=8, spacing3=3)
        self.learning_text.tag_configure("body", spacing3=5)
        self.learning_text.tag_configure("muted", foreground=MUTED, spacing3=5)
        self._set_learning_placeholder()

    def _set_learning_placeholder(self) -> None:
        self.learning_text.configure(state="normal")
        self.learning_text.delete("1.0", "end")
        self.learning_text.insert(
            "end",
            "Nach einem abgeschlossenen Training erscheint hier eine verständliche "
            "Zusammenfassung der verwendeten Beatmaps, ihrer Rhythmen und der "
            "Validierungsergebnisse.\n\nDu kannst auch einen früheren Trainingsbericht öffnen.",
            "muted",
        )
        self.learning_text.configure(state="disabled")

    def _choose_learning_report(self) -> None:
        initialdir = self.last_learning_report.parent if self.last_learning_report else CHECKPOINTS_PATH
        path = filedialog.askopenfilename(
            title="Trainingsbericht auswählen",
            initialdir=str(initialdir),
            filetypes=(("BeatMap-AI Trainingsbericht", "*.learning.json"),
                       ("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")),
        )
        if path:
            self._display_learning_report(Path(path), select_page=True)

    def _load_recent_learning_report(self) -> None:
        if not CHECKPOINTS_PATH.is_dir():
            return
        reports = list(CHECKPOINTS_PATH.glob("*.learning.json"))
        if reports:
            self._display_learning_report(max(reports, key=lambda path: path.stat().st_mtime))

    def _display_learning_report(self, path: Path, *, select_page: bool = False) -> None:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("schema_version") != 1:
                raise ValueError("Dieses Format wird von dieser Version der App nicht unterstützt.")
            title, sections = self._format_learning_report(report)
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            messagebox.showerror("Trainingsbericht konnte nicht geöffnet werden", str(exc), parent=self)
            return

        self.last_learning_report = path.resolve()
        self.learning_report_path_var.set(path.name)
        self.learning_text.configure(state="normal")
        self.learning_text.delete("1.0", "end")
        self.learning_text.insert("end", title + "\n", "title")
        for heading, paragraph, muted in sections:
            self.learning_text.insert("end", heading + "\n", "heading")
            self.learning_text.insert("end", paragraph + "\n\n", "muted" if muted else "body")
        self.learning_text.configure(state="disabled")
        self.open_learning_button.configure(state="normal")
        if select_page:
            self._open_learning_page()

    @staticmethod
    def _format_learning_report(
        report: dict[str, object],
    ) -> tuple[str, list[tuple[str, str, bool]]]:
        maps = int(report.get("maps", 0))
        songs = int(report.get("songs", 0))
        train_maps = int(report.get("training_maps", 0))
        validation_maps = int(report.get("validation_maps", 0))
        note_starts = int(report.get("note_starts", 0))
        slider_starts = int(report.get("slider_starts", 0))
        slider_share = float(report.get("slider_share", 0.0)) * 100.0
        tempo = report.get("tempo_bpm") or {}
        density = report.get("density_objects_per_second") or {}
        validation_f1 = float(report.get("validation_f1", 0.0))
        threshold = float(report.get("decision_threshold", 0.5))

        def grouped(value: int) -> str:
            return f"{value:,}".replace(",", ".")

        def number(value: object, digits: int = 1) -> str:
            if value is None:
                return "nicht verfügbar"
            return f"{float(value):.{digits}f}"

        if tempo.get("median") is None:
            tempo_text = "In den verwendeten Maps ließ sich kein verlässliches Tempo zusammenfassen."
        else:
            tempo_text = (
                f"Typisches Tempo: {number(tempo.get('median'))} BPM; die mittleren 80 % "
                f"der Map-Tempi liegen ungefähr zwischen {number(tempo.get('p10'))} und "
                f"{number(tempo.get('p90'))} BPM."
            )
        if density:
            density_text = (
                f"Typische Notendichte: {number(density.get('median'), 2)} Objekte pro Sekunde "
                f"(mittlere Hälfte der Maps: {number(density.get('p25'), 2)}–"
                f"{number(density.get('p75'), 2)})."
            )
        else:
            density_text = "Für die Notendichte liegen keine Werte vor."

        rhythm_rows = report.get("rhythm_grid", [])
        rhythm_lines = []
        if isinstance(rhythm_rows, list):
            for row in rhythm_rows:
                if not isinstance(row, dict):
                    continue
                share = float(row.get("share", 0.0)) * 100.0
                rhythm_lines.append(f"• {row.get('label', 'Unbekannt')}: {share:.1f} % der Notenstarts")
        rhythm_text = "\n".join(rhythm_lines) if rhythm_lines else "Kein Rhythmusprofil verfügbar."

        created = str(report.get("created_at", ""))
        created_line = f"Erstellt: {created}\n" if created else ""
        return "Trainingsbericht", [
            ("Welche Beatmaps verwendet wurden",
             f"{created_line}{grouped(maps)} verwendbare osu!standard-Schwierigkeiten aus "
             f"{grouped(songs)} Songs. Davon gingen {grouped(train_maps)} Maps ins Training "
             f"und {grouped(validation_maps)} in die Prüfung. Die Beispiele enthielten "
             f"{grouped(note_starts)} Notenstarts.", False),
            ("Rhythmus und Spielweise",
             f"{tempo_text}\n{density_text}\n{grouped(slider_starts)} Starts waren Slider "
             f"({slider_share:.1f} %).", False),
            ("Wo die Noten im Takt liegen",
             rhythm_text,
             True),
            ("Was das Modell daraus lernt",
             "Das Modell verknüpft die Klangmerkmale eines Songs mit dem Zeitpunkt eines "
             "Notenstarts. Zusätzlich berücksichtigt es die Position im Beat und im Takt sowie "
             "die gewünschte Notendichte. Ein zweiter Ausgang schätzt, ob ein Notenstart ein "
             "Slider ist. Die Rhythmusanteile oben beschreiben die Trainingsbeispiele; sie sind "
             "keine Garantie, dass jede erzeugte Map genau diesem Durchschnitt folgt.", False),
            ("Wie gut das Modell die Muster wiedererkennt",
             f"Noten-Erkennungswert auf zurückgehaltenen Songs (F1): {validation_f1:.3f}. "
             f"Der verwendete Entscheidungsschwellenwert ist {threshold:.2f}. Der F1-Wert "
             "misst, wie gut vorhergesagte Notenstarts zu den Vorlagen passen; er bewertet "
             "nicht automatisch, ob eine Map musikalisch Spaß macht.", False),
        ]

    def _open_learning_page(self) -> None:
        self.notebook.select(self.learning_page)

    @staticmethod
    def _labeled_entry(parent: ttk.Frame, label: str, variable: tk.StringVar,
                       row: int, column: int, width: int = 34) -> ttk.Entry:
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.grid(row=row, column=column, sticky="ew", padx=(0 if column == 0 else 14, 0),
                  pady=(0 if row == 0 else 9, 0))
        ttk.Label(cell, text=label, style="CardMuted.TLabel").pack(anchor="w")
        entry = ttk.Entry(cell, textvariable=variable, width=width)
        entry.pack(fill="x", pady=(3, 0))
        parent.grid_columnconfigure(column, weight=1)
        return entry

    def _toggle_generate_advanced(self) -> None:
        self.advanced_generate_open = not self.advanced_generate_open
        if self.advanced_generate_open:
            self.advanced_generate.pack(fill="x", pady=(0, 2))
        else:
            self.advanced_generate.pack_forget()

    def _toggle_train_advanced(self) -> None:
        self.advanced_train_open = not self.advanced_train_open
        if self.advanced_train_open:
            self.advanced_train.pack(fill="x", pady=(8, 0))
        else:
            self.advanced_train.pack_forget()

    def _on_training_mode_change(self, _event=None) -> None:
        mode = self.training_mode_var.get()
        if mode == "Individuell":
            return
        # (epochs, steps per epoch, batch size) sized to the speed of each device:
        # ROCm/CUDA manages ~40 twelve-second excerpts per second, DirectML ~15, CPU ~4.
        sizes = {
            "cuda": {"Kurz": (8, 200, 16), "Ausgewogen": (30, 300, 16)},
            "directml": {"Kurz": (6, 100, 8), "Ausgewogen": (20, 200, 8)},
            "cpu": {"Kurz": (4, 50, 8), "Ausgewogen": (10, 100, 8)},
        }.get(self.detected_backend, {"Kurz": (4, 50, 8), "Ausgewogen": (10, 100, 8)})
        epochs, steps, batch = sizes["Kurz" if mode.startswith("Kurz") else "Ausgewogen"]
        self.epochs_var.set(str(epochs))
        self.steps_var.set(str(steps))
        self.batch_var.set(str(batch))
        # Fine-tuning an existing model needs smaller steps than training a new one.
        self.lr_var.set("0.0003" if self.continue_model_var.get() else "0.001")

    def _refresh_hardware_status(self) -> None:
        backend, description = _detect_backend()
        self.events.put(("hardware", (backend, description)))

    def _update_init_model_state(self) -> None:
        self.init_model_entry.configure(state="normal" if self.continue_model_var.get() else "disabled")
        self._on_training_mode_change()

    def _choose_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Song auswählen", filetypes=(("Audiodateien", "*.mp3 *.ogg"), ("Alle Dateien", "*.*")))
        if path:
            self.audio_var.set(path)
            previous = Path(self.output_map_var.get()).expanduser() if self.output_map_var.get() else None
            output_dir = previous.parent if previous else Path(path).parent
            self.output_map_var.set(str(output_dir / f"{Path(path).stem}.osz"))
            self.open_osu_button.configure(state="disabled")
            self.open_map_folder_button.configure(state="disabled")

    def _choose_map_output(self) -> None:
        initial = self.output_map_var.get() or str(PROJECT_ROOT / "Beatmap.osz")
        path = filedialog.asksaveasfilename(
            title="Beatmap speichern", initialfile=Path(initial).name,
            initialdir=str(Path(initial).parent), defaultextension=".osz",
            filetypes=(("osu! Beatmap-Set", "*.osz"),))
        if path:
            self.output_map_var.set(path)

    def _choose_data_folder(self) -> None:
        initial = self.data_var.get().split(";")[0] or str(Path.home())
        path = filedialog.askdirectory(title="Beatmap-Sammlung auswählen", initialdir=initial)
        if path:
            self.data_var.set(path)

    def _choose_init_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Startmodell auswählen", initialdir=str(MODEL_PATH.parent),
            filetypes=(("PyTorch-Checkpoint", "*.pt"), ("Alle Dateien", "*.*")))
        if path:
            self.init_model_var.set(path)

    def _choose_model_output(self) -> None:
        initial = self.model_output_var.get() or str(CHECKPOINTS_PATH / "rhythm-finetuned.pt")
        path = filedialog.asksaveasfilename(
            title="Neuen Checkpoint speichern", initialfile=Path(initial).name,
            initialdir=str(Path(initial).parent), defaultextension=".pt",
            filetypes=(("PyTorch-Checkpoint", "*.pt"),))
        if path:
            self.model_output_var.set(path)

    def _start_generate(self) -> None:
        if self.active_task:
            return
        audio = Path(self.audio_var.get()).expanduser()
        if not audio.is_file():
            messagebox.showinfo("Song auswählen", "Bitte wähle zuerst eine MP3- oder OGG-Datei aus.", parent=self)
            return
        if audio.suffix.lower() not in (".mp3", ".ogg"):
            messagebox.showinfo("Dateiformat", "Bitte verwende eine MP3- oder OGG-Datei.", parent=self)
            return
        difficulties = [name for name, variable in self.difficulties.items() if variable.get()]
        try:
            stars = [float(part.replace(",", ".")) for part in
                     self.stars_var.get().replace(" ", "").split(";") if part]
        except ValueError:
            messagebox.showerror("Sternzahlen prüfen",
                                 "Trenne mehrere Sternzahlen mit ; – zum Beispiel 3.8; 4.5; 6", parent=self)
            return
        if any(not 0.5 <= value <= 12 for value in stars):
            messagebox.showerror("Sternzahlen prüfen", "Sternzahlen müssen zwischen 0.5 und 12 liegen.", parent=self)
            return
        difficulties += [f"{value:g}" for value in stars]
        if not difficulties:
            messagebox.showinfo("Schwierigkeit auswählen", "Wähle mindestens eine Schwierigkeit aus.", parent=self)
            return
        output_text = self.output_map_var.get().strip()
        output = Path(output_text).expanduser() if output_text else audio.with_suffix(".osz")
        if output.suffix.lower() != ".osz":
            output = output.with_suffix(".osz")
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Speicherort nicht verfügbar", str(exc), parent=self)
            return
        self.output_map_var.set(str(output))
        args = ["generate", str(audio), "-o", str(output)]
        for difficulty in difficulties:
            args.extend(("-d", difficulty))
        if not self.use_model_var.get():
            args.extend(("--no-model", "--rule-placement"))
        args.extend(("--variety", f"{self.variety_var.get() / 100:.2f}"))
        if not self.critic_var.get():
            args.append("--no-critic")
        style =dict(STYLE_CHOICES)[self.style_var.get()]
        if style and self.use_model_var.get():
            args.extend(("--style", f"{style}={self.style_strength_var.get() / 100:.2f}"))
        if self.artist_var.get().strip():
            args.extend(("--artist", self.artist_var.get().strip()))
        if self.title_var.get().strip():
            args.extend(("--title", self.title_var.get().strip()))
        try:
            bpm = self._optional_float(self.bpm_var.get(), "BPM")
            offset = self._optional_float(self.offset_var.get(), "Offset")
        except ValueError as exc:
            messagebox.showerror("Wert prüfen", str(exc), parent=self)
            return
        if bpm is not None:
            if not math.isfinite(bpm) or bpm <= 0:
                messagebox.showerror("BPM prüfen", "BPM muss größer als 0 sein.", parent=self)
                return
            args.extend(("--bpm", str(bpm)))
        if offset is not None:
            if not math.isfinite(offset):
                messagebox.showerror("Offset prüfen", "Der Offset muss eine endliche Zahl sein.", parent=self)
                return
            args.extend(("--offset", str(offset)))
        if output.exists() and not messagebox.askyesno(
                "Datei überschreiben?", f"Diese Datei ist bereits vorhanden:\n{output}\n\nSoll sie ersetzt werden?", parent=self):
            return

        self.last_generated = output
        self.open_osu_button.configure(state="disabled")
        self.open_map_folder_button.configure(state="disabled")
        self._start_job("generate", args, f"Erstelle Beatmap für {audio.name} …")

    def _start_train(self) -> None:
        if self.active_task:
            return
        folders = [Path(p.strip()).expanduser() for p in self.data_var.get().split(";") if p.strip()]
        if not folders or not all(folder.is_dir() for folder in folders):
            messagebox.showinfo("Datenordner auswählen", "Bitte wähle einen vorhandenen osu! Beatmap-Ordner aus.", parent=self)
            return
        output_text = self.model_output_var.get().strip()
        if not output_text:
            messagebox.showinfo("Speicherort auswählen", "Bitte wähle einen Speicherort für das trainierte Modell.", parent=self)
            return
        output = Path(output_text).expanduser()
        if output.suffix.lower() != ".pt":
            output = output.with_suffix(".pt")
        output = output.resolve()
        init_model = Path(self.init_model_var.get()).expanduser().resolve()
        use_init = self.continue_model_var.get()
        if use_init and not init_model.is_file():
            messagebox.showerror("Startmodell fehlt", "Der ausgewählte Start-Checkpoint wurde nicht gefunden.", parent=self)
            return
        if use_init and init_model == output:
            messagebox.showerror("Speicherort prüfen", "Speichere das neue Modell unter einem anderen Namen als das Startmodell.", parent=self)
            return
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Speicherort nicht verfügbar", str(exc), parent=self)
            return
        try:
            epochs = self._positive_int(self.epochs_var.get(), "Epochen")
            steps = self._positive_int(self.steps_var.get(), "Schritte je Epoche")
            batch = self._positive_int(self.batch_var.get(), "Batchgröße")
            workers = self._positive_int(self.workers_var.get(), "CPU-Prozesse")
            lr = self._optional_float(self.lr_var.get(), "Lernrate")
            if lr is None or not math.isfinite(lr) or lr <= 0:
                raise ValueError("Die Lernrate muss größer als 0 sein.")
        except ValueError as exc:
            messagebox.showerror("Trainingseinstellungen prüfen", str(exc), parent=self)
            return
        if output.exists() and not messagebox.askyesno(
                "Checkpoint überschreiben?",
                f"Diese Checkpoint-Datei ist bereits vorhanden:\n{output}\n\nSoll sie ersetzt werden?",
                parent=self):
            return

        choice = self.device_choice_var.get()
        backend = self.detected_backend if choice == "Automatisch" else {
            "DirectML (AMD/Windows)": "directml",
            "CUDA (NVIDIA/ROCm)": "cuda",
            "CPU": "cpu",
        }.get(choice, "cpu")
        args = ["train", *(str(folder.resolve()) for folder in folders), "-o", str(output),
                "--epochs", str(epochs), "--steps-per-epoch", str(steps),
                "--batch-size", str(batch), "--lr", str(lr),
                "--device", backend, "--workers", str(workers)]
        if use_init:
            args.extend(("--init-model", str(init_model)))
        self.model_output_var.set(str(output))
        self.last_checkpoint = output
        self.activate_model_button.configure(state="disabled")
        self.open_checkpoint_folder_button.configure(state="disabled")
        self._start_job("train", args, f"Training startet auf {backend} …", epochs=epochs)

    @staticmethod
    def _optional_float(value: str, label: str) -> float | None:
        value = value.strip().replace(",", ".")
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} muss eine Zahl sein.") from exc

    @staticmethod
    def _positive_int(value: str, label: str) -> int:
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} muss eine ganze Zahl sein.") from exc
        if parsed < 1:
            raise ValueError(f"{label} muss mindestens 1 sein.")
        return parsed

    def _start_job(self, task: str, args: list[str], message: str,
                   *, epochs: int | None = None) -> None:
        self.active_task = task
        self.cancel_requested = False
        self.status_var.set(message)
        self.progress.configure(mode="indeterminate", maximum=epochs or 100, value=0)
        self.progress.start(12)
        self.cancel_button.configure(state="normal")
        self.generate_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        self._append_log(message)
        self.worker = threading.Thread(target=self._run_cli, args=(task, args), daemon=True)
        self.worker.start()

    def _run_cli(self, task: str, args: list[str]) -> None:
        process: subprocess.Popen[str] | None = None
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            process = subprocess.Popen(
                [_console_python(), "-u", "-m", "beatmap_ai", *args],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self.events.put(("process", process))
            assert process.stdout is not None
            for line in process.stdout:
                self.events.put(("log", line.rstrip()))
            code = process.wait()
            self.events.put(("finished", (task, code)))
        except Exception as exc:
            self.events.put(("log", f"Fehler beim Starten: {exc}"))
            self.events.put(("finished", (task, 1)))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    line = str(payload)
                    self._append_log(line)
                    epoch_match = re.search(r"epoch\s+(\d+)\s+loss", line, re.IGNORECASE)
                    if epoch_match:
                        self.progress.stop()
                        self.progress.configure(mode="determinate")
                        self.progress["value"] = int(epoch_match.group(1))
                        total = self.progress["maximum"]
                        self.status_var.set(f"Training läuft – Epoche {epoch_match.group(1)} von {total} …")
                    elif "computing spectrograms" in line.lower():
                        self.status_var.set("Bereite Audio-Merkmale vor …")
                elif kind == "process":
                    self.process = payload  # type: ignore[assignment]
                elif kind == "hardware":
                    self.detected_backend, description = payload  # type: ignore[misc]
                    self.hardware_var.set(description)
                    self._on_training_mode_change()
                elif kind == "finished":
                    task, code = payload  # type: ignore[misc]
                    self._finish_job(str(task), int(code))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _finish_job(self, task: str, code: int) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.process = None
        self.active_task = None
        self.cancel_button.configure(state="disabled")
        self.generate_button.configure(state="normal")
        self.train_button.configure(state="normal")
        if self.cancel_requested:
            self.status_var.set("Aufgabe abgebrochen.")
            self._append_log("Aufgabe wurde abgebrochen.")
        elif code != 0:
            self.status_var.set("Die Aufgabe ist mit einem Fehler beendet worden. Details stehen im Aktivitätsprotokoll.")
            self._append_log(f"Prozess beendet mit Fehlercode {code}.")
        elif task == "generate":
            self.status_var.set("Fertig! Die Beatmap-Datei kann jetzt in osu! importiert werden.")
            self.open_osu_button.configure(state="normal")
            self.open_map_folder_button.configure(state="normal")
            self._append_log("Erstellung abgeschlossen.")
        elif task == "train":
            if self.last_checkpoint and self.last_checkpoint.is_file():
                self.status_var.set("Training fertig. Du kannst das neue Modell jetzt aktivieren.")
                self.activate_model_button.configure(state="normal")
                self.open_checkpoint_folder_button.configure(state="normal")
                self._append_log(f"Checkpoint bereit: {self.last_checkpoint}")
                report_path = self.last_checkpoint.with_suffix(".learning.json")
                if report_path.is_file():
                    self._display_learning_report(report_path, select_page=True)
                    self._append_log(f"Lernübersicht bereit: {report_path}")
                else:
                    self._append_log("Für diesen Trainingslauf wurde kein Lernbericht gefunden.")
            else:
                self.status_var.set("Training beendet, aber es wurde keine Checkpoint-Datei gefunden.")
        else:
            self.status_var.set("Fertig.")

    def _cancel_task(self) -> None:
        if not self.process or not self.active_task:
            return
        if not messagebox.askyesno(
                "Aufgabe abbrechen?",
                "Die laufende Aufgabe wird beendet. Beim Training bleibt der zuletzt gespeicherte Checkpoint erhalten.",
                parent=self):
            return
        self.cancel_requested = True
        self.status_var.set("Aufgabe wird beendet …")
        process = self.process
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                process.terminate()
        except OSError as exc:
            self._append_log(f"Abbruch fehlgeschlagen: {exc}")

    def _activate_model(self) -> None:
        source = self.last_checkpoint
        if not source or not source.is_file():
            messagebox.showerror("Checkpoint fehlt", "Der trainierte Checkpoint wurde nicht gefunden.", parent=self)
            return
        try:
            CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            from .sequence_model import is_sequence_checkpoint
            is_seq = is_sequence_checkpoint(source)
            target = SEQUENCE_PATH if is_seq else MODEL_PATH
            if source.resolve() == target.resolve():
                self.status_var.set("Dieses Modell ist bereits aktiv.")
                return
            if target.exists():
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                kind = "sequence" if is_seq else "rhythm"
                backup = CHECKPOINTS_PATH / f"{kind}-backup-{stamp}.pt"
                shutil.copy2(target, backup)
                self._append_log(f"Bisheriges Standardmodell gesichert: {backup}")
            shutil.copy2(source, target)
            self.activate_model_button.configure(state="disabled")
            self.status_var.set(f"Das trainierte Modell ({'Sequence' if is_seq else 'Rhythmus'}) ist jetzt aktiv.")
            self._append_log(f"Aktives Modell aktualisiert: {target}")
        except OSError as exc:
            messagebox.showerror("Modell konnte nicht aktiviert werden", str(exc), parent=self)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _open_generated(self) -> None:
        if self.last_generated and self.last_generated.is_file():
            try:
                os.startfile(str(self.last_generated))  # type: ignore[attr-defined]
            except OSError as exc:
                messagebox.showerror("Datei konnte nicht geöffnet werden", str(exc), parent=self)

    def _open_generated_folder(self) -> None:
        if self.last_generated:
            self._open_path(self.last_generated.parent)

    def _open_checkpoint_folder(self) -> None:
        path = self.last_checkpoint.parent if self.last_checkpoint else CHECKPOINTS_PATH
        self._open_path(path)

    @staticmethod
    def _open_path(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]

    def _open_help(self) -> None:
        readme = PROJECT_ROOT / "README.md"
        if os.name == "nt" and readme.is_file():
            os.startfile(str(readme))  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Hilfe", f"Kurzanleitung: {readme}", parent=self)

    def _on_close(self) -> None:
        if self.active_task:
            messagebox.showinfo("Aufgabe läuft", "Warte bitte, bis die laufende Aufgabe beendet ist, oder brich sie zuerst ab.", parent=self)
            return
        self.destroy()


def main(initial_audio: str | Path | None = None) -> None:
    app = BeatmapApp(initial_audio=initial_audio)
    app.mainloop()


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    main(audio)
