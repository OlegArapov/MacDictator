import os
import gc
import re
import threading
import tempfile
import subprocess
import logging
import fcntl
import sounddevice as sd
import soundfile as sf
import pyperclip
import pyautogui
import customtkinter as ctk
from openai import OpenAI
import numpy as np
import time
import psutil
import json
from datetime import datetime

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

def _app_data_dir():
    """Return a stable user-writable directory for app state."""
    import sys
    if getattr(sys, 'frozen', False):
        base = os.path.expanduser("~/Library/Application Support/MacDictator")
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        pass
    return base


_DATA_DIR = _app_data_dir()

# --- API KEYS ---
KEYS_FILE = os.path.join(_DATA_DIR, "keys.json")

def _load_keys():
    defaults = {"openai": "", "deepseek": ""}
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception as e:
            logging.warning("Failed to load keys: %s", e)
    return defaults

def _save_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def _make_clients(keys):
    oc = OpenAI(api_key=keys["openai"]) if keys["openai"] else None
    dc = OpenAI(api_key=keys["deepseek"], base_url="https://api.deepseek.com") if keys["deepseek"] else None
    return oc, dc

_api_keys = _load_keys()
openai_client, deepseek_client = _make_clients(_api_keys)

APP_VERSION = "1.0.0"
MAX_RECORD_SEC = 300
LOCK_FILE = os.path.join(_DATA_DIR, ".macdictator.lock")
HISTORY_FILE = os.path.join(_DATA_DIR, "history.json")
SETTINGS_FILE = os.path.join(_DATA_DIR, "settings.json")
PROMPTS_FILE = os.path.join(_DATA_DIR, "prompts.json")

DEFAULTS = {"engine": "MLX", "model": "large-v3", "translate": "Off", "translate_model": "DeepSeek", "cleanup": "Off", "cleanup_model": "DeepSeek", "send": "Off"}

MLX_MODELS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

MLX_MODEL_APPROX_MB = 3072  # large-v3 is ~3 GB

_mlx_lock = threading.Lock()  # prevents concurrent model loads


def _hf_cache_dir(repo):
    return os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}"
    )


def _mlx_model_downloaded(repo):
    """True if HF snapshot for repo is fully cached (no network lookups)."""
    if not mlx_whisper:
        return False
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def _hf_cache_size_mb(repo):
    """Return current cached size in MB for repo (counts snapshot + blobs)."""
    base = _hf_cache_dir(repo)
    if not os.path.isdir(base):
        return 0
    total = 0
    for root, _, files in os.walk(base, followlinks=False):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)

_DEFAULT_PROMPTS = {
    "preamble": (
        "CRITICAL: The text below is a SPEECH-TO-TEXT TRANSCRIPTION (dictation). "
        "The user is NOT talking to you and is NOT asking you a question. "
        "Do NOT reply, do NOT answer, do NOT converse, do NOT add commentary. "
        "Your ONLY job is to edit/clean the dictated text and output the result. "
        "ALWAYS fix punctuation: add commas, periods, question marks where needed. "
        "NEVER use em dashes (—) under any circumstances. Use commas, periods, or colons instead."
    ),
    "cleanup_lite": (
        "You are a punctuation and cleanup assistant. "
        "Remove filler words and speech artifacts: "
        "Russian: ну, вот, типа, короче, значит, блин, ладно, итак, в общем, как бы; "
        "English: um, uh, like, well, so, basically, you know, right, I mean. "
        "Also remove stutters, false starts, and accidental word repetitions. "
        "Do NOT change any wording, meaning, style, sentence structure, or language. "
        "Output ONLY the cleaned text, nothing else."
    ),
    "cleanup_medium": (
        "You are an editor. "
        "1) Remove all filler words, stutters, repetitions, and false starts. "
        "2) Fix grammar errors and sentence structure: fix awkward phrasing, "
        "split run-on sentences, combine fragments. "
        "3) Keep the original meaning, tone, and language. Do not add new information. "
        "Output ONLY the edited text."
    ),
    "cleanup_max": (
        "You are a professional text editor. "
        "Understand what the user INTENDED to say, then rewrite it as clean, "
        "well-structured, polished prose. "
        "Remove all speech artifacts, fillers, and rambling. Fix grammar. "
        "Improve flow, clarity, and readability. Make it sound natural and written, not spoken. "
        "Preserve all key information and the original language. "
        "Output ONLY the rewritten text."
    ),
    "translate": (
        "Professional translator. Translate to {lang}. "
        "Output ONLY the translation, nothing else."
    ),
}

def _load_prompts():
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r") as f:
                saved = json.load(f)
            merged = dict(_DEFAULT_PROMPTS)
            merged.update(saved)
            return merged
        except Exception as e:
            logging.warning("Failed to load prompts: %s", e)
    return dict(_DEFAULT_PROMPTS)

def _save_prompts(prompts):
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

def _build_cleanup_prompts(prompts):
    pre = prompts.get("preamble", "")
    return {
        "Off": None,
        "Lite": pre + prompts.get("cleanup_lite", ""),
        "Medium": pre + prompts.get("cleanup_medium", ""),
        "Max": pre + prompts.get("cleanup_max", ""),
    }

_prompts = _load_prompts()
CLEANUP_PROMPTS = _build_cleanup_prompts(_prompts)

# Map old Russian setting values to new English ones
_SETTINGS_MIGRATION = {
    "model": {"base": "large-v3", "turbo": "large-v3"},
    "cleanup": {"Нет": "Off", "Лайт": "Lite", "Средне": "Medium", "Макс": "Max"},
    "translate": {"Whisper": "→EN", "DeepSeek": "→EN", "ChatGPT": "→EN"},
    "send": {"Нет": "Off", "Да": "Paste", "Вставка": "Paste"},
}

# --- PALETTE ---
C = {
    "bg":       "#101016",
    "card":     "#1A1A24",
    "glass":    "#16161F",
    "hover":    "#252532",
    "border":   "#2E2E3E",
    "border_subtle": "#222230",
    "text":     "#EEEEF2",
    "text2":    "#9999B0",
    "text3":    "#5C5C72",
    "accent":   "#3B82F6",
    "red":      "#DC2626",
    "green":    "#22C55E",
    "orange":   "#F59E0B",
    "pill_bg":  "#1E1E2A",
    "pill_active": "#2A2A3A",
}


class VUMeter(ctk.CTkCanvas):
    """Classic block-based VU meter with green/yellow/red zones."""
    BLOCKS = 12
    BLOCK_W = 8
    BLOCK_H = 5
    GAP = 2
    # zone thresholds (fraction of total blocks)
    GREEN_END = 0.55   # blocks 0-10: green (normal)
    YELLOW_END = 0.80  # blocks 11-15: yellow (quiet/marginal)
    # blocks 16-19: red (clipping)

    def __init__(self, master, **kw):
        total_w = self.BLOCKS * (self.BLOCK_W + self.GAP) - self.GAP
        super().__init__(master, width=total_w, height=self.BLOCK_H,
                         bg=C["bg"], highlightthickness=0, **kw)
        self.total_w = total_w
        self.volume = 0.0
        self.smooth = 0.0
        self.peak = 0.0
        self.peak_decay = 0
        self.active = False
        self._draw_idle()

    def _block_color(self, i, lit):
        frac = i / self.BLOCKS
        if frac < self.GREEN_END:
            return "#51CF66" if lit else "#1A2A1E"
        elif frac < self.YELLOW_END:
            return "#FFA94D" if lit else "#2A251A"
        else:
            return "#FF6B6B" if lit else "#2A1A1A"

    def _draw_idle(self):
        self.delete("all")
        for i in range(self.BLOCKS):
            x = i * (self.BLOCK_W + self.GAP)
            self.create_rectangle(x, 0, x + self.BLOCK_W, self.BLOCK_H,
                                  fill=self._block_color(i, False), outline="")

    def set_volume(self, v):
        self.volume = min(1.0, v)

    def start(self):
        self.active = True
        self.volume = 0.0
        self.smooth = 0.0
        self.peak = 0.0
        self.peak_decay = 0
        self._tick()

    def stop(self):
        self.active = False
        self._progress_active = False
        self._draw_idle()

    def start_progress(self, estimated_seconds):
        """Turn VU meter into a progress bar for estimated_seconds."""
        self.active = False
        self._progress_active = True
        self._progress_start = time.time()
        self._progress_duration = max(estimated_seconds, 1.0)
        self._tick_progress()

    def stop_progress(self):
        self._progress_active = False
        self._draw_idle()

    def _tick_progress(self):
        if not self._progress_active:
            return
        elapsed = time.time() - self._progress_start
        # asymptotic: approaches 95% then slows down
        raw = elapsed / self._progress_duration
        progress = min(0.95, raw) if raw < 1.0 else min(0.99, 0.95 + (raw - 1.0) * 0.02)
        filled = int(progress * self.BLOCKS)
        self.delete("all")
        for i in range(self.BLOCKS):
            x = i * (self.BLOCK_W + self.GAP)
            if i < filled:
                color = "#F59E0B"  # orange for progress
            elif i == filled:
                # animate: blink current block
                color = "#F59E0B" if int(elapsed * 3) % 2 == 0 else "#2A251A"
            else:
                color = "#1A1A1E"
            self.create_rectangle(x, 0, x + self.BLOCK_W, self.BLOCK_H,
                                  fill=color, outline="")
        self.after(100, self._tick_progress)

    def _tick(self):
        if not self.active:
            return
        self.smooth += (self.volume - self.smooth) * 0.35
        # peak hold
        if self.smooth >= self.peak:
            self.peak = self.smooth
            self.peak_decay = 12  # hold for ~400ms
        else:
            if self.peak_decay > 0:
                self.peak_decay -= 1
            else:
                self.peak -= 0.03
                if self.peak < 0:
                    self.peak = 0

        lit_count = int(self.smooth * self.BLOCKS)
        peak_block = min(int(self.peak * self.BLOCKS), self.BLOCKS - 1)

        self.delete("all")
        for i in range(self.BLOCKS):
            x = i * (self.BLOCK_W + self.GAP)
            lit = i < lit_count
            # peak indicator
            if i == peak_block and self.peak > 0.05:
                color = self._block_color(i, True)
            elif lit:
                color = self._block_color(i, True)
            else:
                color = self._block_color(i, False)
            self.create_rectangle(x, 0, x + self.BLOCK_W, self.BLOCK_H,
                                  fill=color, outline="")
        self.after(33, self._tick)


class RecordingBubble(ctk.CTkToplevel):
    """Tiny floating recording indicator — red dot + waveform + timer."""

    _BW, _BH = 120, 24  # compact bubble

    def __init__(self, master):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        self.configure(fg_color="#1A1A24")
        self._volume = 0.0
        self._smooth = 0.0
        self._start_time = 0.0
        self._active = False
        self._wave_history = [0.0] * 24

        self._sw = self.winfo_screenwidth()
        # Start off-screen (no withdraw — avoids activation on show)
        self.geometry(f"{self._BW}x{self._BH}+{-9999}+{-9999}")

        # Single canvas — no frames, no border artifacts
        self._canvas = ctk.CTkCanvas(self, width=self._BW, height=self._BH,
                                     bg="#1A1A24", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        # Set all-spaces immediately (window exists off-screen)
        self.after(100, lambda: self._set_all_spaces_for(self))

    def show(self):
        self._start_time = time.time()
        self._active = True
        self._wave_history = [0.0] * 32
        self._smooth = 0.0
        # Move on-screen (no deiconify — no app activation / space switch)
        self.geometry(f"{self._BW}x{self._BH}+{(self._sw - self._BW) // 2}+6")
        self.lift()
        # Must set all-spaces every show — macOS ignores it for off-screen windows
        self.after(10, lambda: self._set_all_spaces_for(self))
        self._tick()

    def hide(self):
        self._active = False
        # Move off-screen (no withdraw — stays on all spaces)
        self.geometry(f"+{-9999}+{-9999}")

    def set_volume(self, v):
        self._volume = min(1.0, v)

    def _tick(self):
        if not self._active:
            return
        # Smooth volume
        self._smooth += (self._volume - self._smooth) * 0.4
        self._wave_history.append(self._smooth)
        self._wave_history.pop(0)

        c = self._canvas
        w, h = self._BW, self._BH
        c.delete("all")

        # Rounded background
        r = h // 2
        c.create_oval(0, 0, h, h, fill="#1A1A24", outline="#DC2626", width=1)
        c.create_oval(w - h, 0, w, h, fill="#1A1A24", outline="#DC2626", width=1)
        c.create_rectangle(r, 0, w - r, h, fill="#1A1A24", outline="")
        c.create_line(r, 0, w - r, 0, fill="#DC2626")  # top edge
        c.create_line(r, h - 1, w - r, h - 1, fill="#DC2626")  # bottom edge

        # Blinking red dot
        elapsed = time.time() - self._start_time
        dot_on = int(elapsed * 2) % 2 == 0
        dot_color = "#DC2626" if dot_on else "#661010"
        c.create_oval(6, 7, 16, 17, fill=dot_color, outline="")

        # Waveform bars
        bars = len(self._wave_history)
        wave_x0, wave_w = 20, 58
        mid = h // 2
        bar_w = wave_w / bars
        for i, v in enumerate(self._wave_history):
            x = wave_x0 + i * bar_w
            amp = max(1, int(v * mid * 0.85))
            color = "#DC2626" if v > 0.5 else "#22C55E" if v > 0.01 else "#2A2A3A"
            c.create_rectangle(x, mid - amp, x + bar_w - 1, mid + amp,
                               fill=color, outline="")

        # Timer
        em, es = divmod(int(elapsed), 60)
        c.create_text(w - 18, mid, text=f"{em:02d}:{es:02d}",
                      font=("SF Mono", 9, "bold"), fill="#EEEEF2", anchor="center")

        self.after(50, self._tick)

    @staticmethod
    def _set_all_spaces_for(win):
        """Set all-spaces on a specific toplevel window via ctypes."""
        try:
            import ctypes, ctypes.util
            lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
            lib.objc_getClass.restype = ctypes.c_void_p
            lib.objc_getClass.argtypes = [ctypes.c_char_p]
            lib.sel_registerName.restype = ctypes.c_void_p
            lib.sel_registerName.argtypes = [ctypes.c_char_p]
            send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
                ('objc_msgSend', lib))
            send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
                ('objc_msgSend', lib))
            send_idx = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long)(
                ('objc_msgSend', lib))
            send_set = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(
                ('objc_msgSend', lib))

            app = send(lib.objc_getClass(b'NSApplication'),
                       lib.sel_registerName(b'sharedApplication'))
            windows = send(app, lib.sel_registerName(b'windows'))
            count = send_long(windows, lib.sel_registerName(b'count'))
            sel_at = lib.sel_registerName(b'objectAtIndex:')
            sel_set = lib.sel_registerName(b'setCollectionBehavior:')
            behavior = (1 << 0) | (1 << 4)
            for i in range(count):
                w = send_idx(windows, sel_at, i)
                send_set(w, sel_set, behavior)
        except Exception:
            pass


def _seg_button(parent, variable, values, command=None, color=None):
    """Glass-style segmented control: subtle pill bg, colored text for active."""
    active_color = color or C["accent"]
    frame = ctk.CTkFrame(parent, fg_color=C["pill_bg"], corner_radius=8, height=30)
    frame._active_color = active_color
    for val in values:
        btn = ctk.CTkButton(
            frame, text=val, width=0, height=26,
            font=("SF Pro Text", 12), corner_radius=6,
            fg_color="transparent", text_color=C["text3"],
            hover_color=C["hover"],
            command=lambda v=val: (_set_seg(variable, v, frame, values, command)),
        )
        btn.pack(side="left", fill="x", expand=True, padx=2, pady=2)
    _highlight_seg(variable.get(), frame, values)
    return frame


def _set_seg(variable, value, frame, values, command=None):
    variable.set(value)
    _highlight_seg(value, frame, values)
    if command:
        command()


def _highlight_seg(active, frame, values):
    active_color = getattr(frame, '_active_color', C["accent"])
    for btn, val in zip(frame.winfo_children(), values):
        if val == active:
            btn.configure(fg_color=C["pill_active"], text_color=active_color)
        else:
            btn.configure(fg_color="transparent", text_color=C["text3"])


class DictatorApp(ctk.CTk):
    STATE_IDLE = "idle"
    STATE_RECORDING = "recording"
    STATE_PROCESSING = "processing"
    STATE_RESULT = "result"

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.history = self._load_history()
        self._settings = self._load_settings()

        self.title("MacDictator")
        self.configure(fg_color=C["bg"])
        self.resizable(False, False)

        # Position: bottom-center, restore saved position if any
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        pw, ph = 360, 70
        default_x = (sw - pw) // 2
        default_y = sh - ph - 80            # ~80px above Dock
        ox = self._settings.get("overlay_x", default_x)
        oy = self._settings.get("overlay_y", default_y)
        # Clamp to visible screen area (saved pos may be from external monitor)
        if ox < -pw or ox > sw or oy < -ph or oy > sh:
            ox, oy = default_x, default_y
        self.geometry(f"{pw}x{ph}+{ox}+{oy}")

        # Remove title bar — must happen after geometry on macOS
        self.withdraw()
        self.overrideredirect(True)
        self.after(50, self._show_frameless)

        self.app_state = self.STATE_IDLE
        self.audio_data = []
        self.samplerate = 16000
        self.stream = None
        self.rec_start_time = 0
        self.timer_job = None
        self.countdown_job = None
        self.cancelled = False
        self._processing_lock = threading.Lock()
        self._psutil_proc = psutil.Process(os.getpid())

        # Set saved mic device before opening stream
        self._selected_device = None
        saved_mic = self._settings.get("mic")
        if saved_mic and saved_mic != "System Default":
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    short = d['name'][:28] + ".." if len(d['name']) > 30 else d['name']
                    if short == saved_mic:
                        self._selected_device = i
                        break
        # if not found or "System Default" → _selected_device stays None (sounddevice uses system default)

        # track active mic name for poll
        try:
            if self._selected_device is None:
                self._active_mic_name = sd.query_devices(kind='input')['name']
            else:
                self._active_mic_name = sd.query_devices(self._selected_device)['name']
        except Exception:
            self._active_mic_name = ""

        self._build_ui()
        self._rec_bubble = RecordingBubble(self)
        self.bind("<Button-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag_motion)
        self.bind("<ButtonRelease-1>", self._on_drag_end)
        self._on_translate_change()   # enforce model lock if Whisper translate loaded from settings
        self._on_cleanup_change()     # show/hide cleanup model row
        self._on_engine_change()      # show/hide model row
        self._ui_ready = True
        self._update_mode_label()
        self._init_stream()  # keep mic always open for instant start
        if self._mic_live:
            self.vu.start()  # show live VU meter
        self.start_keyboard_listener()
        self._update_resources()
        self._preload_model()
        self.after(5000, self._poll_default_mic)

    # --- persistence ---
    def _load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    s = {**DEFAULTS, **json.load(f)}
                # migrate old Russian values to English
                for key, mapping in _SETTINGS_MIGRATION.items():
                    if key in s and s[key] in mapping:
                        s[key] = mapping[s[key]]
                if s.get("mic") == "_default":
                    s["mic"] = "System Default"
                return s
        except Exception as e:
            logging.warning("Failed to load settings: %s", e)
        return dict(DEFAULTS)

    def _save_settings(self):
        try:
            data = {
                "engine": self.engine_var.get(),
                "model": self.model_var.get(),
                "translate": self.translate_var.get(),
                "translate_model": self.translate_model_var.get(),
                "cleanup": self.cleanup_var.get(),
                "cleanup_model": self.cleanup_model_var.get(),
                "send": self.send_var.get(),
                "mic": "_default" if self.mic_var.get() == "System Default" else self.mic_var.get(),
                "overlay_x": self.winfo_x(),
                "overlay_y": self.winfo_y(),
            }
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._update_mode_label()
            self._update_indicators()
        except Exception as e:
            logging.warning("Failed to save settings: %s", e)

    def _load_history(self):
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logging.warning("Failed to load history: %s", e)
        return []

    def _save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history[-10:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning("Failed to save history: %s", e)

    # --- UI ---
    def _build_ui(self):
        """Build compact pill UI."""
        # Outer pill frame — rounded, draggable
        pill = ctk.CTkFrame(self, fg_color=C["card"], corner_radius=0,
                            border_width=2, border_color=C["border"])
        pill.pack(fill="both", expand=True, padx=3, pady=3)
        self._pill = pill  # keep reference for border color changes

        # Bind drag to pill and all its children
        pill.bind("<Button-1>", self._on_drag_start)
        pill.bind("<B1-Motion>", self._on_drag_motion)
        pill.bind("<ButtonRelease-1>", self._on_drag_end)

        # Inner container — keeps content away from border
        inner = ctk.CTkFrame(pill, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=4, pady=4)

        # Left: mic icon + status
        left = ctk.CTkFrame(inner, fg_color="transparent")
        left.pack(side="left", padx=(4, 2))

        self._rec_indicator = ctk.CTkLabel(
            left, text="●", font=("SF Pro Text", 18), width=20,
            text_color=C["red"])
        self._rec_indicator.pack(side="left")

        info_col = ctk.CTkFrame(left, fg_color="transparent")
        info_col.pack(side="left", padx=(4, 0))

        self.status_label = ctk.CTkLabel(
            info_col, text="Ready", height=14,
            font=("SF Pro Text", 10), text_color=C["text3"], anchor="w")
        self.status_label.pack(anchor="w")

        # mic name — resolve active mic
        try:
            if self._selected_device is None:
                _mic_display = sd.query_devices(kind='input')['name']
            else:
                _mic_display = sd.query_devices(self._selected_device)['name']
        except Exception:
            _mic_display = "Unknown"
        _mic_short = _mic_display[:22] + ".." if len(_mic_display) > 24 else _mic_display

        self._mic_name_label = ctk.CTkLabel(
            info_col, text=_mic_short, height=11,
            font=("SF Pro Text", 8), text_color=C["text3"], anchor="w")
        self._mic_name_label.pack(anchor="w")

        # hotkey hint (⌘R) — shown in idle state
        self._hotkey_label = ctk.CTkLabel(
            info_col, text="⌘R", height=10,
            font=("SF Mono", 8), text_color=C["text3"], anchor="w")
        self._hotkey_label.pack(anchor="w")

        # Center column: indicators + VU meter
        center = ctk.CTkFrame(inner, fg_color="transparent")
        center.pack(side="left", fill="both", expand=True, padx=(4, 2))

        # Top row: small status indicators
        indicators = ctk.CTkFrame(center, fg_color="transparent")
        indicators.pack(fill="x", pady=(0, 0))

        s = self._settings
        model_text = "OpenAI" if "OpenAI" in s.get("engine", "MLX") else "MLX"
        send_text = s.get("send", "Off")
        cleanup_text = s.get("cleanup", "Off")

        self._ind_model = ctk.CTkLabel(
            indicators, text=model_text, height=10,
            font=("SF Mono", 8, "bold"), text_color="#6366F1")
        self._ind_model.pack(side="left", padx=(0, 3))

        translate_text = s.get("translate", "Off")
        lang_text = translate_text if translate_text not in ("Off", "Whisper", "DeepSeek", "ChatGPT") else "RU"
        self._ind_lang = ctk.CTkLabel(
            indicators, text=lang_text, height=10,
            font=("SF Mono", 8, "bold"), text_color="#0EA5E9")
        self._ind_lang.pack(side="left", padx=(0, 3))

        if cleanup_text != "Off":
            self._ind_cleanup = ctk.CTkLabel(
                indicators, text=f"✨{cleanup_text}", height=10,
                font=("SF Mono", 8), text_color="#F59E0B")
            self._ind_cleanup.pack(side="left", padx=(0, 6))
        else:
            self._ind_cleanup = None

        send_color = C["green"] if send_text != "Off" else C["text3"]
        self._ind_send = ctk.CTkLabel(
            indicators, text=f"→{send_text}" if send_text != "Off" else "→Off", height=10,
            font=("SF Mono", 8), text_color=send_color)
        self._ind_send.pack(side="left", padx=(0, 3))

        # Timer label (on same indicators row, right side)
        self.timer_label = ctk.CTkLabel(
            indicators, text="", height=10, width=0,
            font=("SF Mono", 9, "bold"), text_color=C["text"])
        self.timer_label.pack(side="right")

        # VU meter below indicators
        self.vu = VUMeter(center)
        self.vu.pack(fill="x", pady=(1, 3))
        self.vu.bind("<Button-1>", self._on_drag_start)
        self.vu.bind("<B1-Motion>", self._on_drag_motion)
        self.vu.bind("<ButtonRelease-1>", self._on_drag_end)

        # Right side: gear + LIVE
        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.pack(side="right", padx=(0, 2))

        self._gear_btn = ctk.CTkButton(
            right, text="⚙", width=28, height=24,
            font=("SF Pro Text", 16), corner_radius=4,
            fg_color="transparent", text_color=C["text"],
            hover_color=C["hover"],
            command=self._open_settings_popup)
        self._gear_btn.pack(pady=(0, 2))

        self._mic_always_on = True
        self._mic_live = True
        self.mic_btn = ctk.CTkButton(
            right, text="LIVE", width=38, height=18,
            font=("SF Mono", 8, "bold"), corner_radius=3,
            fg_color="transparent", text_color=C["green"],
            hover_color=C["hover"],
            command=self._toggle_mic_mode)
        self.mic_btn.pack()

        # Start VU meter

        # init state vars needed by rest of app
        self._panels_visible = False
        self._sound_on = True
        self._pinned = True

        # Stub vars that old code references (settings popup will set real values)
        self._init_setting_vars()

    def _init_setting_vars(self):
        s = self._settings
        self.engine_var = ctk.StringVar(value=s.get("engine", "MLX"))
        self.model_var = ctk.StringVar(value="large-v3")
        self.translate_var = ctk.StringVar(value=s.get("translate", "Off"))
        self.translate_model_var = ctk.StringVar(value=s.get("translate_model", "DeepSeek"))
        self.cleanup_var = ctk.StringVar(value=s.get("cleanup", "Off"))
        self.cleanup_model_var = ctk.StringVar(value=s.get("cleanup_model", "DeepSeek"))
        self.send_var = ctk.StringVar(value=s.get("send", "Paste"))
        saved_mic = s.get("mic", "System Default")
        self.mic_var = ctk.StringVar(value=saved_mic)
        self._input_devices = self._get_input_devices()

    def _update_indicators(self):
        """Refresh pill indicator labels from current settings vars."""
        if not hasattr(self, '_ind_model'):
            return
        engine = self.engine_var.get()
        model = "OpenAI" if "OpenAI" in engine else "MLX"
        self._ind_model.configure(text=model)

        translate = self.translate_var.get()
        lang_text = translate if translate != "Off" else "RU"
        lang_color = "#0EA5E9" if translate != "Off" else "#0EA5E9"
        self._ind_lang.configure(text=lang_text, text_color=lang_color)

        send = self.send_var.get()
        send_color = C["green"] if send != "Off" else C["text3"]
        self._ind_send.configure(text=f"→{send}", text_color=send_color)

        cleanup = self.cleanup_var.get()
        if cleanup != "Off":
            if self._ind_cleanup is None:
                self._ind_cleanup = ctk.CTkLabel(
                    self._ind_send.master, text=f"✨{cleanup}", height=12,
                    font=("SF Mono", 9), text_color="#F59E0B")
                # pack before send indicator
                self._ind_cleanup.pack(side="left", padx=(0, 6),
                                       before=self._ind_send)
            else:
                self._ind_cleanup.configure(text=f"✨{cleanup}")
        else:
            if self._ind_cleanup is not None:
                self._ind_cleanup.pack_forget()
                self._ind_cleanup.destroy()
                self._ind_cleanup = None

    def _open_settings_popup(self):
        if hasattr(self, '_settings_win') and self._settings_win and self._settings_win.winfo_exists():
            self._settings_win.focus()
            return

        win = ctk.CTkToplevel(self)
        self._settings_win = win
        win.title("MacDictator Settings")
        win.geometry("420x560")
        win.resizable(True, True)
        win.configure(fg_color=C["bg"])
        win.attributes('-topmost', True)

        tabs = ctk.CTkTabview(win, fg_color=C["bg"],
                              segmented_button_fg_color=C["card"],
                              segmented_button_selected_color=C["accent"],
                              segmented_button_selected_hover_color="#2563EB",
                              segmented_button_unselected_color=C["card"],
                              segmented_button_unselected_hover_color=C["hover"],
                              text_color=C["text"])
        tabs.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        tabs.add("Settings")
        tabs.add("History")

        stab = tabs.tab("Settings")
        self._build_settings_tab(stab)

        htab = tabs.tab("History")
        self._build_history_tab(htab)

        footer = ctk.CTkFrame(win, fg_color="transparent")
        footer.pack(fill="x", padx=12, pady=(8, 12))
        ctk.CTkButton(footer, text="API Keys & Prompts", width=160, height=30,
                      font=("SF Pro Text", 12), fg_color=C["hover"],
                      text_color=C["text2"], hover_color=C["border"],
                      command=self._open_keys_window).pack(side="left")
        ctk.CTkLabel(footer, text=f"v{APP_VERSION}",
                     font=("SF Mono", 9), text_color=C["text3"]).pack(side="right")

    def _build_settings_tab(self, parent):
        def _save_wrap(cmd=None):
            if cmd:
                cmd()
            self._save_settings()

        _LBL_W = 48
        _LBL_FONT = ("SF Pro Text", 11)

        def _row(par, label, variable, values, cmd=None, pady=0, color=None):
            r = ctk.CTkFrame(par, fg_color="transparent")
            ctk.CTkLabel(r, text=label, width=_LBL_W, font=_LBL_FONT,
                         text_color=C["text3"], anchor="e").pack(side="left", padx=(12, 6))
            seg = _seg_button(r, variable, values, command=lambda: _save_wrap(cmd), color=color)
            seg.pack(side="left", fill="x", expand=True, padx=(0, 12))
            r.pack(fill="x", pady=pady)
            return r, seg

        card = ctk.CTkFrame(parent, fg_color=C["glass"], corner_radius=12,
                            border_width=1, border_color=C["border_subtle"])
        card.pack(fill="x", padx=2, pady=(6, 0))

        # Mic row
        mic_row = ctk.CTkFrame(card, fg_color="transparent")
        ctk.CTkLabel(mic_row, text="Mic", width=_LBL_W, font=_LBL_FONT,
                     text_color=C["text3"], anchor="e").pack(side="left", padx=(12, 6))
        device_names = ["System Default"] + [d["name_short"] for d in self._input_devices]
        if self.mic_var.get() not in device_names:
            device_names.append(self.mic_var.get())
        self._mic_menu = ctk.CTkOptionMenu(
            mic_row, variable=self.mic_var, values=device_names,
            height=28, font=("SF Pro Text", 12),
            fg_color=C["pill_bg"], button_color=C["pill_active"],
            button_hover_color=C["hover"], text_color=C["text2"],
            dropdown_fg_color=C["glass"], dropdown_text_color=C["text"],
            dropdown_hover_color=C["hover"], corner_radius=8,
            command=lambda v: self._on_mic_change())
        self._mic_menu.pack(side="left", fill="x", expand=True, padx=(0, 12))
        mic_row.pack(fill="x", pady=(10, 8))

        # Engine (MLX = large-v3 local, OpenAI = cloud)
        _row(card, "Engine", self.engine_var, ["MLX", "OpenAI"],
             self._on_engine_change, pady=(0, 4), color="#7C7CFF")

        ctk.CTkFrame(card, fg_color=C["border_subtle"], height=1).pack(fill="x", padx=16, pady=(4, 8))

        translate_val = self.translate_var.get()
        if translate_val not in ("Off", "→EN", "→RU"):
            # migrate old values: "Whisper"/"DeepSeek"/"ChatGPT" → "→EN"
            if translate_val != "Off":
                translate_val = "→EN"
            else:
                translate_val = "Off"
            self.translate_var.set(translate_val)
        _row(card, "Translate", self.translate_var,
             ["Off", "→EN", "→RU"], self._on_translate_change,
             pady=(0, 2), color="#5BA8D6")

        translate_model_val = self.translate_model_var.get()
        if translate_model_val not in ("DeepSeek", "ChatGPT"):
            translate_model_val = "DeepSeek"
            self.translate_model_var.set(translate_model_val)
        self._translate_model_row = ctk.CTkFrame(card, fg_color="transparent")
        ctk.CTkLabel(self._translate_model_row, text="via", width=_LBL_W,
                     font=("SF Pro Text", 10), text_color=C["text3"],
                     anchor="e").pack(side="left", padx=(12, 6))
        _seg_button(self._translate_model_row, self.translate_model_var,
                     ["DeepSeek", "ChatGPT"], command=lambda: _save_wrap(),
                     color="#5BA8D6").pack(side="left", fill="x", expand=True, padx=(0, 12))
        self._translate_model_row.pack(fill="x", pady=(0, 5))
        self._on_translate_change()

        self._cleanup_row, _ = _row(card, "Cleanup", self.cleanup_var,
                                    ["Off", "Lite", "Medium", "Max"],
                                    self._on_cleanup_change, pady=0, color="#D4A054")

        cleanup_model_val = self.cleanup_model_var.get()
        if cleanup_model_val not in ("DeepSeek", "ChatGPT"):
            cleanup_model_val = "DeepSeek"
            self.cleanup_model_var.set(cleanup_model_val)
        self._cleanup_model_row = ctk.CTkFrame(card, fg_color="transparent")
        ctk.CTkLabel(self._cleanup_model_row, text="via", width=_LBL_W,
                     font=("SF Pro Text", 10), text_color=C["text3"],
                     anchor="e").pack(side="left", padx=(12, 6))
        via_seg = _seg_button(self._cleanup_model_row, self.cleanup_model_var,
                              ["DeepSeek", "ChatGPT"], command=lambda: _save_wrap(),
                              color="#D4A054")
        via_seg.pack(side="left", fill="x", expand=True, padx=(0, 12))
        if self.cleanup_var.get() != "Off":
            self._cleanup_model_row.pack(fill="x", pady=(4, 0))
        else:
            self._cleanup_model_row.pack_forget()

        _row(card, "Send", self.send_var, ["Off", "Paste", "Enter"],
             pady=(8, 10), color="#4ADE80")

    def _build_history_tab(self, parent):
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=(4, 0))
        ctk.CTkLabel(top, text="History", font=("SF Pro Text", 13, "bold"),
                     text_color=C["text3"]).pack(side="left")
        ctk.CTkButton(top, text="Clear", width=60, height=23,
                      font=("SF Pro Text", 12), fg_color="transparent",
                      text_color=C["text3"], hover_color=C["hover"],
                      command=self._clear_history).pack(side="right")

        self.resource_label = ctk.CTkLabel(
            parent, text="", height=15,
            font=("SF Mono", 9), text_color=C["text3"])
        self.resource_label.pack(side="bottom", pady=(4, 0))

        self.history_frame = ctk.CTkScrollableFrame(
            parent, fg_color="transparent",
            scrollbar_button_color=C["border"])
        self.history_frame.pack(fill="both", expand=True, padx=2)
        self._rebuild_history()

    def _update_mode_label(self):
        pass  # pill has no mode indicators — settings visible in popup

    def _draw_rec_button(self, recording=False, processing=False):
        pass  # pill uses _rec_indicator label instead

    def _toggle_panels(self):
        pass  # replaced by _open_settings_popup

    def _open_keys_window(self):
        if hasattr(self, '_keys_win') and self._keys_win and self._keys_win.winfo_exists():
            self._keys_win.focus()
            return

        win = ctk.CTkToplevel(self)
        self._keys_win = win
        win.title("Setup")
        win.geometry("500x560")
        win.resizable(True, True)
        win.configure(fg_color=C["bg"])
        win.attributes('-topmost', True)

        pad = 12
        keys = _load_keys()
        prompts = _load_prompts()
        entries = {}
        status_labels = {}
        prompt_boxes = {}

        # --- Tabs ---
        tabs = ctk.CTkTabview(win, fg_color=C["bg"], segmented_button_fg_color=C["card"],
                              segmented_button_selected_color=C["accent"],
                              segmented_button_selected_hover_color="#2563EB",
                              segmented_button_unselected_color=C["card"],
                              segmented_button_unselected_hover_color=C["hover"],
                              text_color=C["text"])
        tabs.pack(fill="both", expand=True, padx=pad, pady=(pad, 0))
        tabs.add("API Keys")
        tabs.add("Prompts")

        # ===== TAB: API Keys =====
        keys_tab = tabs.tab("API Keys")

        for label, key_name in [("OpenAI", "openai"), ("DeepSeek", "deepseek")]:
            frame = ctk.CTkFrame(keys_tab, fg_color=C["card"], corner_radius=8)
            frame.pack(fill="x", pady=(4, 0))

            top = ctk.CTkFrame(frame, fg_color="transparent")
            top.pack(fill="x", padx=8, pady=(6, 0))

            ctk.CTkLabel(top, text=label, font=("SF Pro Text", 14, "bold"),
                         text_color=C["text"]).pack(side="left")

            st_lbl = ctk.CTkLabel(top, text="—", font=("SF Pro Text", 13),
                                  text_color=C["text3"], width=78)
            st_lbl.pack(side="right", padx=(4, 0))
            status_labels[key_name] = st_lbl

            check_btn = ctk.CTkButton(
                top, text="Test", width=47, height=23,
                font=("SF Mono", 10, "bold"), corner_radius=3,
                fg_color=C["hover"], text_color=C["text2"],
                hover_color=C["border"])
            check_btn.pack(side="right", padx=(4, 0))

            show_var = ctk.BooleanVar(value=False)
            toggle_btn = ctk.CTkButton(
                top, text="👁", width=31, height=23,
                font=("SF Pro Text", 13), corner_radius=3,
                fg_color="transparent", text_color=C["text3"],
                hover_color=C["hover"])
            toggle_btn.pack(side="right")

            entry = ctk.CTkEntry(frame, font=("SF Mono", 13), height=36,
                                 fg_color=C["bg"], border_color=C["border"],
                                 text_color=C["text"], show="•")
            entry.pack(fill="x", padx=8, pady=(4, 8))
            entry.insert(0, keys.get(key_name, ""))
            entries[key_name] = entry

            def _make_toggle(e=entry, v=show_var, b=toggle_btn):
                def _toggle():
                    v.set(not v.get())
                    e.configure(show="" if v.get() else "•")
                    b.configure(text_color=C["text"] if v.get() else C["text3"])
                return _toggle
            toggle_btn.configure(command=_make_toggle())

            def _make_check(kn=key_name, e=entry, sl=st_lbl, cb=check_btn):
                def _check():
                    key_val = e.get().strip()
                    if not key_val:
                        sl.configure(text="no key", text_color=C["orange"])
                        return
                    sl.configure(text="...", text_color=C["orange"])
                    cb.configure(state="disabled")
                    def _do_check():
                        try:
                            if kn == "openai":
                                c = OpenAI(api_key=key_val)
                                c.models.list()
                            else:
                                c = OpenAI(api_key=key_val, base_url="https://api.deepseek.com")
                                c.models.list()
                            win.after(0, lambda: sl.configure(text="OK", text_color=C["green"]))
                        except Exception:
                            win.after(0, lambda: sl.configure(text="Error", text_color=C["red"]))
                        finally:
                            win.after(0, lambda: cb.configure(state="normal"))
                    threading.Thread(target=_do_check, daemon=True).start()
                return _check
            check_btn.configure(command=_make_check())

        # MLX Whisper status
        mlx_frame = ctk.CTkFrame(keys_tab, fg_color=C["card"], corner_radius=8)
        mlx_frame.pack(fill="x", pady=(8, 0))
        mlx_top = ctk.CTkFrame(mlx_frame, fg_color="transparent")
        mlx_top.pack(fill="x", padx=8, pady=(6, 8))
        ctk.CTkLabel(mlx_top, text="MLX Whisper", font=("SF Pro Text", 14, "bold"),
                     text_color=C["text"]).pack(side="left")

        default_repo = MLX_MODELS["large-v3"]

        def _mlx_current_status():
            if not mlx_whisper:
                return ("Not installed", C["red"])
            if _mlx_model_downloaded(default_repo):
                return ("Ready", C["green"])
            return ("Model not downloaded", C["orange"])

        _txt, _col = _mlx_current_status()
        mlx_status_lbl = ctk.CTkLabel(mlx_top, text=_txt, font=("SF Pro Text", 13),
                                      text_color=_col)
        mlx_status_lbl.pack(side="right")

        mlx_action_frame = ctk.CTkFrame(mlx_frame, fg_color="transparent")
        if mlx_whisper and not _mlx_model_downloaded(default_repo):
            mlx_action_frame.pack(fill="x", padx=8, pady=(0, 8))

            dl_btn = ctk.CTkButton(mlx_action_frame,
                                   text=f"Download model (~{MLX_MODEL_APPROX_MB // 1024} GB)",
                                   height=32, fg_color=C["accent"])
            dl_btn.pack(fill="x")

            def _start_download():
                dl_btn.configure(state="disabled")
                state = {"done": False, "error": None}

                def _bg():
                    try:
                        from huggingface_hub import snapshot_download
                        snapshot_download(default_repo)
                    except Exception as e:
                        state["error"] = str(e)
                    finally:
                        state["done"] = True

                threading.Thread(target=_bg, daemon=True).start()

                def _tick():
                    if state["done"]:
                        if state["error"]:
                            dl_btn.configure(state="normal", text="Retry download")
                            mlx_status_lbl.configure(text="Download failed", text_color=C["red"])
                        else:
                            mlx_action_frame.pack_forget()
                            mlx_status_lbl.configure(text="Ready", text_color=C["green"])
                        return
                    mb = _hf_cache_size_mb(default_repo)
                    pct = min(99, int(mb * 100 / MLX_MODEL_APPROX_MB))
                    dl_btn.configure(text=f"Downloading... {pct}%  ({mb:.0f} / {MLX_MODEL_APPROX_MB} MB)")
                    dl_btn.after(500, _tick)

                _tick()

            dl_btn.configure(command=_start_download)

        # Microphone status
        mic_frame = ctk.CTkFrame(keys_tab, fg_color=C["card"], corner_radius=8)
        mic_frame.pack(fill="x", pady=(4, 0))
        mic_top = ctk.CTkFrame(mic_frame, fg_color="transparent")
        mic_top.pack(fill="x", padx=8, pady=(6, 8))
        ctk.CTkLabel(mic_top, text="Microphone", font=("SF Pro Text", 14, "bold"),
                     text_color=C["text"]).pack(side="left")
        try:
            dev = sd.query_devices(kind='input')
            mic_name = dev['name'][:30]
            mic_ok = True
        except Exception:
            mic_name = "Not found"
            mic_ok = False
        ctk.CTkLabel(mic_top, text=mic_name, font=("SF Pro Text", 13),
                     text_color=C["green"] if mic_ok else C["red"]).pack(side="right")

        # ===== TAB: Prompts =====
        prompts_tab = tabs.tab("Prompts")

        scroll = ctk.CTkScrollableFrame(prompts_tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        prompt_fields = [
            ("preamble",      "Preamble (добавляется ко всем cleanup)",  80,  C["text3"]),
            ("cleanup_lite",  "Cleanup · Lite",                          80,  C["orange"]),
            ("cleanup_medium","Cleanup · Medium",                        100, C["orange"]),
            ("cleanup_max",   "Cleanup · Max",                           100, C["orange"]),
            ("translate",     "Перевод  ({lang} = язык назначения)",      50,  C["accent"]),
        ]

        for key, label, height, color in prompt_fields:
            ctk.CTkLabel(scroll, text=label, font=("SF Mono", 11, "bold"),
                         text_color=color, anchor="w").pack(fill="x", pady=(10, 2))
            tb = ctk.CTkTextbox(scroll, height=height, font=("SF Pro Text", 12),
                                fg_color=C["card"], text_color=C["text"],
                                border_width=1, border_color=C["border"],
                                corner_radius=4, wrap="word")
            tb.pack(fill="x")
            tb.insert("1.0", prompts.get(key, _DEFAULT_PROMPTS.get(key, "")))
            prompt_boxes[key] = tb

        # ===== Save / Cancel =====
        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=pad, pady=(6, pad))

        def _save():
            global openai_client, deepseek_client, _api_keys, _prompts, CLEANUP_PROMPTS
            new_keys = {
                "openai": entries["openai"].get().strip(),
                "deepseek": entries["deepseek"].get().strip(),
            }
            _save_keys(new_keys)
            _api_keys = new_keys
            openai_client, deepseek_client = _make_clients(new_keys)

            new_prompts = {k: tb.get("1.0", "end").strip() for k, tb in prompt_boxes.items()}
            _save_prompts(new_prompts)
            _prompts = new_prompts
            CLEANUP_PROMPTS = _build_cleanup_prompts(new_prompts)
            win.destroy()

        ctk.CTkButton(btn_frame, text="Save", height=39,
                       font=("SF Pro Text", 16, "bold"),
                       fg_color=C["accent"], hover_color="#2563EB",
                       corner_radius=6, command=_save).pack(side="right")
        ctk.CTkButton(btn_frame, text="Cancel", height=39,
                       font=("SF Pro Text", 16),
                       fg_color="transparent", text_color=C["text2"],
                       hover_color=C["hover"],
                       corner_radius=6, command=win.destroy).pack(side="right", padx=(0, 8))

    def _toggle_pin(self):
        pass  # pill is always on top

    def _toggle_sound(self):
        self._sound_on = not self._sound_on  # keep logic, button gone

    def _show_frameless(self):
        self.deiconify()
        self.attributes('-topmost', True)
        self.after(100, self._set_all_spaces)
        self.after(200, self._setup_tray)

    def _set_all_spaces(self):
        """Make window visible on all macOS desktops/spaces via ctypes (no pyobjc)."""
        try:
            import ctypes, ctypes.util
            lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
            lib.objc_getClass.restype = ctypes.c_void_p
            lib.objc_getClass.argtypes = [ctypes.c_char_p]
            lib.sel_registerName.restype = ctypes.c_void_p
            lib.sel_registerName.argtypes = [ctypes.c_char_p]
            # Different call signatures for objc_msgSend
            send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
                ('objc_msgSend', lib))
            send_long_ret = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
                ('objc_msgSend', lib))
            send_idx = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long)(
                ('objc_msgSend', lib))
            send_set = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(
                ('objc_msgSend', lib))

            app = send(lib.objc_getClass(b'NSApplication'),
                       lib.sel_registerName(b'sharedApplication'))
            windows = send(app, lib.sel_registerName(b'windows'))
            count = send_long_ret(windows, lib.sel_registerName(b'count'))
            sel_at = lib.sel_registerName(b'objectAtIndex:')
            sel_set = lib.sel_registerName(b'setCollectionBehavior:')
            # canJoinAllSpaces (1<<0) | stationary (1<<4)
            behavior = (1 << 0) | (1 << 4)
            for i in range(count):
                win = send_idx(windows, sel_at, i)
                send_set(win, sel_set, behavior)
        except Exception as e:
            logging.warning("Failed to set all-spaces: %s", e)

    def _setup_tray(self):
        """Launch menu bar tray icon as a separate process (rumps)."""
        import sys
        if getattr(sys, 'frozen', False):
            return  # sys.executable points to the app bundle, not a usable Python
        try:
            tray_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tray.py")
            if os.path.exists(tray_script):
                self._tray_proc = subprocess.Popen(
                    [sys.executable, tray_script, str(os.getpid())])
        except Exception as e:
            logging.warning("Tray: %s", e)

    def _toggle_visibility(self):
        visible = self.winfo_viewable()
        if visible:
            self.withdraw()
        else:
            self.overrideredirect(True)
            self.deiconify()
            self.attributes('-topmost', True)
            self.lift()
            self.after(50, self._set_all_spaces)

    def _quit_app(self):
        # Kill tray subprocess if running
        if hasattr(self, '_tray_proc') and self._tray_proc:
            try:
                self._tray_proc.terminate()
            except Exception:
                pass
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.destroy()

    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag_motion(self, event):
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        x = self.winfo_x() + dx
        y = self.winfo_y() + dy
        self.geometry(f"+{x}+{y}")

    def _on_drag_end(self, event):
        # save position
        self._settings["overlay_x"] = self.winfo_x()
        self._settings["overlay_y"] = self.winfo_y()
        self._save_settings()

    def _toggle_mic_mode(self):
        self._mic_live = not self._mic_live
        if self._mic_live:
            self.mic_btn.configure(text="LIVE", text_color=C["green"])
            self._mic_always_on = True
            self._open_mic()
            if self._mic_timer:
                self.after_cancel(self._mic_timer)
                self._mic_timer = None
            if self.app_state == self.STATE_IDLE:
                self.vu.start()
        else:
            self.mic_btn.configure(text="MIC", text_color=C["text3"])
            self._mic_always_on = False
            self._reset_mic_timer()
            if self.app_state == self.STATE_IDLE:
                self.vu.stop()

    def _get_input_devices(self):
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                name = d['name']
                short = name[:28] + ".." if len(name) > 30 else name
                devices.append({"index": i, "name": name, "name_short": short})
        return devices

    def _on_mic_change(self):
        self._save_settings()
        # Restart mic stream with new device
        sel = self.mic_var.get()
        dev_idx = None
        for d in self._input_devices:
            if d["name_short"] == sel:
                dev_idx = d["index"]
                break
        self._selected_device = dev_idx
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._open_mic()
        self.update_status(f"Mic: {sel}", "green")

    def _poll_default_mic(self):
        """Auto-switch to system default mic if no explicit mic is pinned."""
        try:
            if getattr(self, 'mic_var', None) and self.mic_var.get() == "System Default":
                current_default = sd.query_devices(kind='input')['name']
                active_name = getattr(self, '_active_mic_name', None)
                if active_name != current_default:
                    self._active_mic_name = current_default
                    self._selected_device = None
                    self._close_mic()
                    self._open_mic()
                    # update mic label in pill if it exists
                    if hasattr(self, '_mic_name_label'):
                        short = current_default[:22] + ".." if len(current_default) > 24 else current_default
                        self._mic_name_label.configure(text=short)
            # Check pinned (non-default) device is still available
            elif getattr(self, '_selected_device', None) is not None:
                try:
                    dev_info = sd.query_devices(self._selected_device)
                    if dev_info['max_input_channels'] == 0:
                        raise RuntimeError("Device lost input channels")
                except Exception:
                    logging.warning("Pinned mic device unavailable, falling back to system default")
                    self._selected_device = None
                    self._active_mic_name = sd.query_devices(kind='input')['name']
                    self._close_mic()
                    self._open_mic()
                    if hasattr(self, '_mic_name_label'):
                        short = self._active_mic_name[:22] + ".." if len(self._active_mic_name) > 24 else self._active_mic_name
                        self._mic_name_label.configure(text=short)
                    self.after(0, lambda: self.update_status("Mic switched to default", "orange"))
        except Exception as e:
            logging.warning("Mic poll error: %s", e)
        self.after(5000, self._poll_default_mic)

    def _on_model_change(self):
        if self.app_state == self.STATE_IDLE:
            self._preload_model()


    def _on_cleanup_change(self):
        if not hasattr(self, '_cleanup_model_row'):
            return
        if self.cleanup_var.get() == "Off":
            self._cleanup_model_row.pack_forget()
        else:
            self._cleanup_model_row.pack(fill="x", pady=(4, 0),
                                         after=self._cleanup_row)

    def _on_translate_change(self):
        if hasattr(self, '_translate_model_row'):
            if self.translate_var.get() == "Off":
                self._translate_model_row.pack_forget()
            else:
                self._translate_model_row.pack(fill="x", pady=(0, 5))
        self._save_settings()
        if hasattr(self, '_update_indicators'):
            self._update_indicators()

    def _on_engine_change(self):
        if getattr(self, '_ui_ready', False):
            self._preload_model()

    def _update_resources(self):
        try:
            if not hasattr(self, 'resource_label') or not self.resource_label.winfo_exists():
                self.after(2000, self._update_resources)
                return
            app_mem = self._psutil_proc.memory_info().rss / 1024 / 1024
            sys_mem = psutil.virtual_memory()
            total_gb = sys_mem.total / 1024 / 1024 / 1024
            used_gb = sys_mem.used / 1024 / 1024 / 1024
            cpu = self._psutil_proc.cpu_percent(interval=None)
            self.resource_label.configure(
                text=f"{app_mem:.0f}MB  |  {used_gb:.1f}/{total_gb:.0f}GB  |  CPU {cpu:.0f}%")
        except Exception as e:
            logging.warning("Failed to update resources: %s", e)
        self.after(2000, self._update_resources)

    # --- status ---
    def update_status(self, msg, color="gray"):
        colors = {
            "gray": C["text3"], "red": C["red"], "green": C["green"],
            "blue": C["accent"], "orange": C["orange"],
        }
        self.status_label.configure(
            text=msg, text_color=colors.get(color, C["text3"]))
        self.update()

    # --- timer ---
    def _start_timer(self):
        self.rec_start_time = time.time()
        self._tick_timer()

    def _tick_timer(self):
        if self.app_state != self.STATE_RECORDING:
            return
        elapsed = int(time.time() - self.rec_start_time)
        em, es = divmod(elapsed, 60)
        remaining = MAX_RECORD_SEC - elapsed
        self.timer_label.configure(
            text=f"{em:02d}:{es:02d}",
            text_color=C["red"] if remaining < 30 else C["text"])
        if elapsed >= MAX_RECORD_SEC:
            self.update_status("Limit", "orange")
            self.stop_and_process()
            return
        self.timer_job = self.after(500, self._tick_timer)

    def _stop_timer(self):
        if self.timer_job:
            self.after_cancel(self.timer_job)
            self.timer_job = None
        self.timer_label.configure(text="")

    # --- cancel ---
    def _cancel(self):
        if self.app_state == self.STATE_RECORDING:
            self.app_state = self.STATE_IDLE
            self._recording = False
            self.vu.stop()
            self._stop_timer()
            self.audio_data = []
            self._to_idle("Cancelled")
        elif self.app_state == self.STATE_PROCESSING:
            self.cancelled = True
            self._to_idle("Cancelled")
        elif self.app_state == self.STATE_RESULT:
            if self.countdown_job:
                self.after_cancel(self.countdown_job)
                self.countdown_job = None
            self._to_idle("Ready")

    def _to_idle(self, msg="Ready", color="gray"):
        self.app_state = self.STATE_IDLE
        self.vu.stop_progress()
        self._rec_bubble.hide()
        self._rec_indicator.configure(text="●", text_color=C["red"])
        self._pill.configure(border_color=C["border"])
        self._hotkey_label.configure(text="⌘R")
        self.update_status(msg, color)
        # Resume live VU if enabled
        if self._mic_live:
            self.vu.start()

    # --- audio ---
    def start_keyboard_listener(self):
        from pynput import keyboard as kb
        from pynput._util.darwin import keycode_context
        import contextlib

        # Cache keyboard context on MAIN THREAD to avoid HIToolbox crash.
        # pynput's Listener._run() calls keycode_context() from a background
        # thread, which triggers TSMGetInputSourceProperty off the main queue.
        # We grab the context here and monkeypatch _run() to reuse it.
        with keycode_context() as ctx:
            cached_context = (ctx[0], ctx[1])  # (keyboard_type, layout_data)

        _orig_run = kb.Listener._run

        def _patched_run(self_listener):
            @contextlib.contextmanager
            def _cached_keycode_context():
                yield cached_context

            # Temporarily replace keycode_context in the module
            import pynput.keyboard._darwin as _mod
            orig_kc = _mod.keycode_context
            _mod.keycode_context = _cached_keycode_context
            try:
                _orig_run(self_listener)
            finally:
                _mod.keycode_context = orig_kc

        kb.Listener._run = _patched_run

        pressed = set()

        def on_press(key):
            pressed.add(key)
            if (kb.Key.ctrl_l in pressed or kb.Key.ctrl_r in pressed) and \
               (kb.Key.shift_l in pressed or kb.Key.shift_r in pressed) and \
               key == kb.Key.space:
                self.after(0, self.toggle_recording)
            elif key == kb.Key.cmd_r:
                self.after(0, self.toggle_recording)
            elif key == kb.Key.esc:
                self.after(0, self._cancel)

        def on_release(key):
            pressed.discard(key)

        def _run_listener():
            with kb.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()

        threading.Thread(target=_run_listener, daemon=True).start()

    def toggle_recording(self):
        if self.app_state == self.STATE_IDLE:
            self.start_recording()
        elif self.app_state == self.STATE_RECORDING:
            self.stop_and_process()
        elif self.app_state == self.STATE_RESULT:
            self.start_recording()

    def _beep(self, count=1):
        if not self._sound_on:
            return
        for i in range(count):
            try:
                subprocess.Popen(
                    ['afplay', '/System/Library/Sounds/Tink.aiff'],
                    close_fds=True,
                    env={"PATH": "/usr/bin:/bin"}
                )
            except Exception as e:
                logging.warning("Failed to play beep: %s", e)
            if i < count - 1:
                time.sleep(0.15)

    def _preload_model(self):
        if not mlx_whisper:
            return
        engine = getattr(self, 'engine_var', None)
        if engine and "OpenAI" in engine.get():
            return
        if not engine and "OpenAI" in self._settings.get("engine", "MLX"):
            return
        model_var = getattr(self, 'model_var', None)
        model_name = model_var.get() if model_var else self._settings.get("model", "large-v3")
        repo = MLX_MODELS.get(model_name)
        if not repo:
            return
        def _load():
            if not _mlx_lock.acquire(blocking=False):
                return  # another load already in progress
            try:
                if not _mlx_model_downloaded(repo):
                    return  # skip preload; download is user-initiated from Setup
                self.after(0, lambda: self.update_status(f"Loading {model_name}...", "orange"))
                import mlx.core as mx
                from mlx_whisper.transcribe import ModelHolder
                if ModelHolder.model_path != repo:
                    ModelHolder.model = None
                    ModelHolder.model_path = None
                    gc.collect()
                ModelHolder.get_model(repo, mx.float16)
                self.after(0, lambda: self.update_status("Ready", "green"))
            except Exception:
                self.after(0, lambda: self.update_status("Ready", "green"))
            finally:
                _mlx_lock.release()
        threading.Thread(target=_load, daemon=True).start()

    def _init_stream(self):
        """Open microphone stream. Auto-closes after idle timeout, reopens on record."""
        self._recording = False
        self._stream_error = False
        self._mic_idle_timeout = 30  # seconds
        self._mic_timer = None
        self._open_mic()

    def _open_mic(self):
        if self.stream and self.stream.active:
            return
        dev = getattr(self, '_selected_device', None)
        try:
            self.stream = sd.InputStream(samplerate=self.samplerate, channels=1,
                                         device=dev,
                                         callback=self._audio_always_callback)
            self.stream.start()
            self._reset_mic_timer()
        except Exception as e:
            logging.warning("Failed to open microphone: %s", e)
            self.stream = None
            self.after(0, lambda: self.update_status("Mic error", "red"))

    def _close_mic(self):
        if self._recording:
            return  # don't close during recording
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _reset_mic_timer(self):
        if self._mic_timer:
            self.after_cancel(self._mic_timer)
            self._mic_timer = None
        if not self._mic_always_on:
            self._mic_timer = self.after(
                self._mic_idle_timeout * 1000, self._close_mic)

    def _audio_always_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning("Audio stream status: %s", status)
            self._stream_error = True
        rms = float(np.sqrt(np.mean(indata ** 2)))
        if self._recording:
            self.audio_data.append(indata.copy())
            self.vu.set_volume(min(1.0, rms * 5.0))
            self._rec_bubble.set_volume(min(1.0, rms * 5.0))
        elif self._mic_live:
            self.vu.set_volume(min(1.0, rms * 5.0))

    def start_recording(self):
        self.app_state = self.STATE_RECORDING
        self.cancelled = False
        self.audio_data = []
        # Force reopen mic if stream had errors (device disconnect, etc.)
        if self._stream_error:
            logging.info("Reopening mic due to prior stream error")
            self._close_mic()
            self._stream_error = False
        self._open_mic()  # reopen if closed
        self._recording = True
        self._rec_indicator.configure(text="◉", text_color=C["red"])
        self._pill.configure(border_color=C["red"])
        self._hotkey_label.configure(text="")
        self.update_status("Recording", "red")
        self.vu.start()
        self._rec_bubble.show()
        self._start_timer()
        threading.Thread(target=lambda: self._beep(2), daemon=True).start()

    def stop_and_process(self):
        if self.app_state != self.STATE_RECORDING:
            return
        self.app_state = self.STATE_PROCESSING
        self._recording = False
        self._reset_mic_timer()  # restart idle countdown
        self.vu.stop()
        self._rec_bubble.hide()
        threading.Thread(target=lambda: self._beep(1), daemon=True).start()
        self._stop_timer()
        self._rec_indicator.configure(text="·····", text_color=C["orange"])
        self._pill.configure(border_color=C["orange"])
        # Estimate processing time: ~0.7x of recording duration for large-v3
        rec_duration = time.time() - self.rec_start_time if self.rec_start_time else 5.0
        estimated = max(rec_duration * 0.7, 2.0)
        self.vu.start_progress(estimated)
        self.update_status("Whisper...", "orange")
        # Lower window so user can click on target app
        threading.Thread(target=self.process_audio, daemon=True).start()

    def _vad_split(self, audio, sr):
        """Split audio at silence boundaries. Returns list of numpy arrays."""
        MIN_CHUNK_SEC = 5
        MAX_CHUNK_SEC = 28
        FRAME_SIZE = 400       # 25ms at 16kHz
        HOP_SIZE = 160         # 10ms at 16kHz
        MIN_SILENCE_FRAMES = 30  # 300ms

        total_samples = len(audio)
        max_chunk_samples = MAX_CHUNK_SEC * sr
        min_chunk_samples = MIN_CHUNK_SEC * sr

        # Short audio — no splitting needed
        if total_samples <= max_chunk_samples * 1.2:
            return [audio]

        # Compute frame-level RMS energy
        n_frames = max(1, (total_samples - FRAME_SIZE) // HOP_SIZE + 1)
        frame_energy = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * HOP_SIZE
            end = min(start + FRAME_SIZE, total_samples)
            frame = audio[start:end].flatten()
            frame_energy[i] = np.sqrt(np.mean(frame ** 2))

        # Adaptive silence threshold based on noise floor
        noise_floor = np.percentile(frame_energy, 10)
        threshold = max(noise_floor * 3, 0.005)

        is_silence = frame_energy < threshold

        # Find silence gap centers (gaps > MIN_SILENCE_FRAMES)
        silence_centers = []  # (center_frame_idx, gap_length)
        i = 0
        while i < n_frames:
            if is_silence[i]:
                gap_start = i
                while i < n_frames and is_silence[i]:
                    i += 1
                gap_len = i - gap_start
                if gap_len >= MIN_SILENCE_FRAMES:
                    center = gap_start + gap_len // 2
                    silence_centers.append((center, gap_len))
            else:
                i += 1

        # Convert frame indices to sample indices
        split_points = [c * HOP_SIZE for c, _ in silence_centers]

        # Build chunks respecting min/max constraints
        chunks = []
        chunk_start = 0
        for sp in split_points:
            if sp - chunk_start < min_chunk_samples:
                continue  # chunk too short, skip this split
            if sp - chunk_start > max_chunk_samples:
                # Need to force-split: find lowest energy point in last 5s
                search_start = max(chunk_start, sp - 5 * sr) // HOP_SIZE
                search_end = sp // HOP_SIZE
                if search_start < search_end:
                    min_idx = search_start + np.argmin(frame_energy[search_start:search_end])
                    force_sp = min_idx * HOP_SIZE
                    if force_sp - chunk_start >= min_chunk_samples:
                        chunks.append(audio[chunk_start:force_sp])
                        chunk_start = force_sp
            if sp > chunk_start and sp - chunk_start >= min_chunk_samples:
                chunks.append(audio[chunk_start:sp])
                chunk_start = sp

        # Add remaining audio
        if total_samples - chunk_start >= sr * 1:  # at least 1s
            chunks.append(audio[chunk_start:])

        # If no splits found, force-split at max intervals
        if not chunks:
            for start in range(0, total_samples, max_chunk_samples):
                end = min(start + max_chunk_samples, total_samples)
                if end - start >= sr * 1:
                    chunks.append(audio[start:end])

        logging.info("VAD split: %d chunks, durations: %s",
                     len(chunks), [f"{len(c)/sr:.1f}s" for c in chunks])
        return chunks

    def _clean_hallucination(self, text):
        """Detect and trim Whisper hallucination patterns. Returns clean text or raises."""
        _HALLUCINATION_PHRASES = [
            "продолжение следует", "субтитры", "редактор субтитров", "корректор",
            "подписывайтесь", "ставьте лайк", "thank you", "thanks for watching",
            "subscribe", "like and subscribe", "смотрите в следующей серии",
            "конец фильма", "музыка",
        ]
        text_lower = text.lower().strip()

        # Short text entirely matching a known hallucination phrase
        for phrase in _HALLUCINATION_PHRASES:
            if phrase in text_lower and len(text) < 80:
                raise Exception("Whisper hallucination — speak louder or check mic")

        # CJK hallucination
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
        if cjk_chars > len(text) * 0.3:
            raise Exception("Whisper hallucination (CJK)")

        # Detect repeating n-grams and truncate at the repetition boundary
        words = text.split()
        if len(words) > 8:
            # Check bigrams and trigrams for repetition loops
            for n in (2, 3, 1):
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                # Sliding window: find where a pattern repeats 4+ times in a row
                i = 0
                while i < len(ngrams):
                    pattern = ngrams[i]
                    if len(pattern) < 2:
                        i += 1
                        continue
                    repeat_count = 1
                    j = i + n
                    while j < len(ngrams) and ngrams[j] == pattern:
                        repeat_count += 1
                        j += n
                    if repeat_count >= 4:
                        # Found hallucination loop — truncate before it
                        cut_word_idx = i
                        clean_text = ' '.join(words[:cut_word_idx]).strip()
                        # Remove trailing hallucination phrase fragments
                        for phrase in _HALLUCINATION_PHRASES:
                            if clean_text.lower().endswith(phrase):
                                clean_text = clean_text[:-(len(phrase))].strip()
                        logging.warning("Hallucination trimmed at word %d/%d, pattern=%r×%d",
                                        cut_word_idx, len(words), pattern, repeat_count)
                        if len(clean_text.split()) < 3:
                            raise Exception("Whisper hallucination (repetitive)")
                        return clean_text
                    i += 1

            # Global check: any single word > 40% of total
            from collections import Counter
            most_common_word, count = Counter(words).most_common(1)[0]
            if count > len(words) * 0.4:
                logging.warning("Hallucination detected: word=%r count=%d/%d", most_common_word, count, len(words))
                raise Exception("Whisper hallucination (repetitive)")

        return text

    def process_audio(self):
        if not self._processing_lock.acquire(blocking=False):
            logging.warning("process_audio already running, skipping")
            return
        filename = os.path.join(tempfile.gettempdir(), "macdictator_temp_audio.wav")
        try:
            if self.cancelled:
                return
            if not self.audio_data:
                raise Exception("Recording too short")
            audio_concat = np.concatenate(self.audio_data, axis=0)
            duration = len(audio_concat) / self.samplerate
            if duration < 0.5:
                raise Exception("Recording too short")
            # Check audio level — reject silence before sending to Whisper
            rms = float(np.sqrt(np.mean(audio_concat ** 2)))
            logging.info("Recording: duration=%.1fs, rms=%.5f, chunks=%d", duration, rms, len(self.audio_data))
            if rms < 0.003:
                raise Exception("Too quiet — speak louder or check mic")
            sf.write(filename, audio_concat, self.samplerate)

            dm, ds = divmod(int(duration), 60)
            self.after(0, lambda: self.update_status(
                f"Whisper {dm:02d}:{ds:02d}...", "orange"))

            engine = self.engine_var.get()
            model_name = self.model_var.get()
            translate_to = self.translate_var.get()  # "Off", "→EN", "→RU"
            translate_model = self.translate_model_var.get()  # "DeepSeek", "ChatGPT"
            cleanup_level = self.cleanup_var.get()
            cleanup_model = self.cleanup_model_var.get()

            if self.cancelled:
                return

            text = ""
            # --- transcribe ---

            if "OpenAI" in engine:
                if not openai_client:
                    raise Exception("OpenAI API key not set (KEY)")
                try:
                    with open(filename, "rb") as f:
                        resp = openai_client.audio.transcriptions.create(
                            model="whisper-1", file=f)
                    text = resp.text.strip()
                except Exception as e:
                    err = str(e)
                    if "insufficient_quota" in err or "429" in err:
                        raise Exception("No credits on OpenAI account")
                    raise
            else:
                repo = MLX_MODELS.get(model_name)
                if not repo:
                    raise Exception(f"Unknown model: {model_name}")

                if not _mlx_model_downloaded(repo):
                    raise Exception("Model not downloaded. Open Setup → Download model")

                # If translating →RU, user speaks English
                if translate_to == "→RU":
                    _lang = "en"
                    _prompt = "This is English speech."
                else:
                    _lang = "ru"
                    _prompt = "Это русская речь."
                _whisper_kwargs = dict(
                    condition_on_previous_text=False,
                    no_speech_threshold=0.3,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    initial_prompt=_prompt,
                    word_timestamps=True,
                    hallucination_silence_threshold=2.0,
                )

                # Split long audio at silence boundaries to prevent hallucination
                chunks = self._vad_split(audio_concat, self.samplerate)

                all_texts = []
                for ci, chunk in enumerate(chunks):
                    if self.cancelled:
                        return
                    if len(chunks) > 1:
                        self.after(0, lambda i=ci, n=len(chunks):
                                   self.update_status(f"Whisper {i+1}/{n}...", "orange"))
                    chunk_file = os.path.join(tempfile.gettempdir(), f"macdictator_chunk_{ci}.wav")
                    sf.write(chunk_file, chunk, self.samplerate)

                    chunk_dur = len(chunk) / self.samplerate
                    timeout_sec = max(120, int(chunk_dur * 10))
                    _result_box = [None, None]

                    def _do_transcribe(_f=chunk_file):
                        try:
                            with _mlx_lock:
                                r = mlx_whisper.transcribe(
                                    _f, path_or_hf_repo=repo,
                                    language=_lang, **_whisper_kwargs)
                            _result_box[0] = r
                        except Exception as e:
                            _result_box[1] = e

                    t = threading.Thread(target=_do_transcribe, daemon=True)
                    t.start()
                    t.join(timeout=timeout_sec)
                    if t.is_alive():
                        raise Exception(f"Transcription timed out on chunk {ci+1}")
                    if _result_box[1]:
                        raise _result_box[1]
                    if _result_box[0] is None:
                        continue
                    # Filter segments by quality metrics
                    segments = _result_box[0].get('segments', [])
                    good_texts = []
                    for seg in segments:
                        cr = seg.get('compression_ratio', 0)
                        lp = seg.get('avg_logprob', 0)
                        nsp = seg.get('no_speech_prob', 0)
                        if cr > 2.4 or lp < -1.0 or nsp > 0.6:
                            logging.info("Filtered segment: cr=%.1f lp=%.2f nsp=%.2f text=%r",
                                         cr, lp, nsp, seg.get('text', '')[:60])
                            continue
                        good_texts.append(seg.get('text', '').strip())
                    chunk_text = ' '.join(good_texts).strip()
                    if not chunk_text:
                        chunk_text = _result_box[0]['text'].strip()
                    if chunk_text:
                        try:
                            chunk_text = self._clean_hallucination(chunk_text)
                            all_texts.append(chunk_text)
                        except Exception:
                            logging.warning("Chunk %d hallucinated, skipping", ci)

                    # Cleanup temp chunk file
                    try:
                        os.unlink(chunk_file)
                    except Exception:
                        pass

                text = ' '.join(all_texts)

            if self.cancelled:
                return
            if not text:
                raise Exception("Empty result")
            logging.info("Whisper raw text: %r", text[:200])

            # Detect and trim Whisper hallucination
            text = self._clean_hallucination(text)

            # Build steps for history
            model_label = "OpenAI Whisper" if "OpenAI" in engine else model_name
            transcribe_label = f"Транскрипция · {model_label}"
            steps = [{"label": transcribe_label, "text": text}]

            cleanup_prompt = CLEANUP_PROMPTS.get(cleanup_level)
            if cleanup_prompt:
                self.after(0, lambda l=cleanup_level, m=cleanup_model:
                           self.update_status(f"Cleanup {l} ({m})...", "orange"))
                try:
                    text = self._llm_call(cleanup_prompt, text, cleanup_model)
                    steps.append({"label": f"Cleanup {cleanup_level} · {cleanup_model}", "text": text})
                except Exception as e:
                    steps.append({"label": f"Cleanup FAILED", "text": str(e)[:80]})
                    self.after(0, lambda: self.update_status("Cleanup failed, saved raw", "orange"))

            if self.cancelled:
                return

            # --- translate via LLM ---
            if translate_to != "Off":
                target_lang = "English" if translate_to == "→EN" else "Russian"
                self.after(0, lambda d=translate_to, m=translate_model:
                           self.update_status(f"{d} ({m})...", "orange"))
                prompt = _prompts.get("translate", _DEFAULT_PROMPTS["translate"]).replace("{lang}", target_lang)
                try:
                    text = self._llm_call(prompt, text, translate_model)
                    steps.append({"label": f"Перевод {translate_to} · {translate_model}", "text": text})
                except Exception as e:
                    steps.append({"label": f"Перевод FAILED", "text": str(e)[:80]})
                    self.after(0, lambda: self.update_status("Translate failed, saved raw", "orange"))

            if self.cancelled:
                return

            self._clipboard_copy(text)
            time.sleep(0.15)  # let clipboard settle before paste
            # Paste using pyautogui (Quartz, no subprocess fork, safe from background thread)
            send_mode = self.send_var.get()
            try:
                if send_mode == "Enter":
                    pyautogui.hotkey('command', 'v')
                    time.sleep(0.3)
                    pyautogui.press('return')
                elif send_mode == "Paste":
                    pyautogui.hotkey('command', 'v')
                # "Off" = copy only, no paste
            except Exception as e:
                logging.warning("Failed to paste: %s", e)
            # Then update UI
            self.after(0, lambda t=text, st=steps: self._on_result(t, st))

        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)[:80]
            if not self.cancelled:
                self.after(0, lambda m=msg: self._on_error(m))
        finally:
            self._processing_lock.release()
            if os.path.exists(filename):
                os.remove(filename)

    def _on_error(self, msg):
        self._to_idle(f"Error: {msg}", "red")

    def _on_result(self, text, steps=None):
        self.app_state = self.STATE_RESULT
        self.vu.stop_progress()
        self._add_history_entry(text, steps)
        # Show preview in pill
        self._rec_indicator.configure(text="✓", text_color=C["green"])
        self._pill.configure(border_color=C["green"])
        preview = text[:20] + "…" if len(text) > 20 else text
        self.update_status(preview, "green")
        self._hotkey_label.configure(text="")
        # Auto-dismiss after 3 seconds
        self.after(3000, self._dismiss_result)

    def _dismiss_result(self):
        if self.app_state == self.STATE_RESULT:
            self._to_idle("Ready", "gray")

    def _deepseek(self, system_prompt, text):
        if not deepseek_client:
            raise Exception("DeepSeek API key not set (KEY)")
        try:
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "insufficient_quota" in err or "429" in err:
                raise Exception("No credits on DeepSeek account")
            raise

    def _llm_call(self, prompt, text, model_name):
        if model_name == "ChatGPT":
            if not openai_client:
                raise Exception("OpenAI API key not set (KEY)")
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                if "insufficient_quota" in err or "429" in err:
                    raise Exception("No credits on OpenAI account")
                raise
        else:
            return self._deepseek(prompt, text)

    # --- history ---
    def _add_history_entry(self, text, steps=None):
        entry = {
            "text": text,
            "time": datetime.now().strftime("%H:%M"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "steps": steps or [],
        }
        self.history.append(entry)
        self.history = self.history[-10:]
        self._save_history()
        self._rebuild_history()

    def _rebuild_history(self):
        """Clear and rebuild all history widgets from data."""
        if not hasattr(self, 'history_frame') or not self.history_frame.winfo_exists():
            return
        for w in self.history_frame.winfo_children():
            w.destroy()
        # show all history, newest first
        for entry in reversed(self.history):
            self._add_history_widget(entry)

    def _add_history_widget(self, entry):
        frame = ctk.CTkFrame(self.history_frame, fg_color=C["card"],
                             corner_radius=6, border_width=1, border_color=C["border"])
        frame.pack(fill="x", pady=(0, 4))

        full_text = entry["text"]
        t = entry.get("time", "")
        steps = entry.get("steps", [])
        display = full_text if len(full_text) <= 50 else full_text[:47] + "..."

        # compact row: time + preview (click to expand)
        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x")

        ctk.CTkButton(
            top, text="C", width=24, height=22,
            font=("SF Mono", 10), corner_radius=4,
            fg_color="transparent", text_color=C["text3"],
            hover_color=C["green"],
            command=lambda: self._copy_from_history(full_text),
        ).pack(side="left", padx=(4, 0), pady=2)

        ctk.CTkLabel(top, text=t, font=("SF Mono", 10),
                     text_color=C["text3"], width=39).pack(side="left", padx=(4, 4), pady=3)

        lbl = ctk.CTkLabel(top, text=display, font=("SF Pro Text", 13),
                           text_color=C["text2"], anchor="w", cursor="hand2")
        lbl.pack(side="left", fill="x", expand=True, pady=3, padx=(0, 4))

        # expand/collapse panel (hidden by default)
        sep = ctk.CTkFrame(frame, height=1, fg_color=C["border"])
        detail = ctk.CTkFrame(frame, fg_color=C["card"], corner_radius=0)
        detail._visible = False

        # show processing steps
        for step in steps:
            color = C["red"] if "FAILED" in step["label"] else C["text3"]
            ctk.CTkLabel(detail, text=step["label"],
                         font=("SF Mono", 10, "bold"), text_color=color,
                         anchor="w").pack(fill="x", padx=6, pady=(6, 1))
            tb = ctk.CTkTextbox(detail, height=60, font=("SF Pro Text", 12),
                                fg_color=C["card"], text_color=C["text"],
                                border_width=1, border_color=C["border"],
                                corner_radius=4)
            tb.pack(fill="x", padx=6, pady=(0, 0))
            tb.insert("1.0", step["text"])
            self._add_resize_handle(detail, tb)

        # final editable textbox (last step = final result)
        if not steps:
            ctk.CTkLabel(detail, text="raw", font=("SF Mono", 10, "bold"),
                         text_color=C["text3"], anchor="w").pack(fill="x", padx=6, pady=(6, 1))
        textbox = ctk.CTkTextbox(detail, height=78, font=("SF Pro Text", 14),
                                  fg_color=C["card"], text_color=C["text"],
                                  border_width=1, border_color=C["border"],
                                  corner_radius=4)
        if steps:
            final_row = ctk.CTkFrame(detail, fg_color="transparent")
            final_row.pack(fill="x", padx=6, pady=(4, 1))
            ctk.CTkLabel(final_row, text="final",
                         font=("SF Mono", 10, "bold"), text_color=C["green"],
                         anchor="w").pack(side="left")
            ctk.CTkButton(
                final_row, text="Copy", width=42, height=20,
                font=("SF Pro Text", 10), corner_radius=4,
                fg_color=C["border"], text_color=C["text3"],
                hover_color=C["green"],
                command=lambda: self._copy_from_history(textbox.get("1.0", "end").strip()),
            ).pack(side="right")
        textbox.pack(fill="x", padx=6, pady=(0, 0))
        textbox.insert("1.0", full_text)
        self._add_resize_handle(detail, textbox)

        # action buttons row
        btn_row = ctk.CTkFrame(detail, fg_color="transparent")
        btn_row.pack(fill="x", padx=4, pady=(0, 4))

        ctk.CTkButton(
            btn_row, text="Copy", width=70, height=29,
            font=("SF Pro Text", 12), corner_radius=4,
            fg_color=C["border"], text_color=C["text2"],
            hover_color=C["accent"],
            command=lambda: self._copy_from_history(textbox.get("1.0", "end").strip()),
        ).pack(side="left", padx=(0, 2))

        ctk.CTkButton(
            btn_row, text="Lite", width=46, height=29,
            font=("SF Pro Text", 12), corner_radius=4,
            fg_color=C["border"], text_color=C["orange"],
            hover_color=C["orange"],
            command=lambda: self._rewrite_in_place(textbox, "Lite"),
        ).pack(side="left", padx=(0, 2))

        ctk.CTkButton(
            btn_row, text="Med", width=46, height=29,
            font=("SF Pro Text", 12), corner_radius=4,
            fg_color=C["border"], text_color=C["orange"],
            hover_color=C["orange"],
            command=lambda: self._rewrite_in_place(textbox, "Medium"),
        ).pack(side="left", padx=(0, 2))

        ctk.CTkButton(
            btn_row, text="Max", width=46, height=29,
            font=("SF Pro Text", 12), corner_radius=4,
            fg_color=C["border"], text_color=C["orange"],
            hover_color=C["orange"],
            command=lambda: self._rewrite_in_place(textbox, "Max"),
        ).pack(side="left", padx=(0, 2))

        ctk.CTkButton(
            btn_row, text="→ RU", width=55, height=29,
            font=("SF Pro Text", 12), corner_radius=4,
            fg_color=C["border"], text_color=C["accent"],
            hover_color=C["accent"],
            command=lambda: self._translate_in_place(textbox, "Russian"),
        ).pack(side="right", padx=(2, 0))

        ctk.CTkButton(
            btn_row, text="→ EN", width=55, height=29,
            font=("SF Pro Text", 12), corner_radius=4,
            fg_color=C["border"], text_color=C["accent"],
            hover_color=C["accent"],
            command=lambda: self._translate_in_place(textbox, "English"),
        ).pack(side="right")

        def toggle(e=None):
            if detail._visible:
                sep.pack_forget()
                detail.pack_forget()
                detail._visible = False
            else:
                sep.pack(fill="x", after=top)
                detail.pack(fill="x", after=sep)
                detail._visible = True

        lbl.bind("<Button-1>", toggle)
        top.bind("<Button-1>", toggle)

    def _add_resize_handle(self, parent, textbox):
        """Drag handle below textbox — pull down to increase height."""
        handle = ctk.CTkFrame(parent, height=7, fg_color=C["border"],
                              cursor="sb_v_double_arrow", corner_radius=2)
        handle.pack(fill="x", padx=12, pady=(0, 4))
        # three-dot grip indicator
        grip = ctk.CTkLabel(handle, text="· · ·", font=("SF Pro Text", 8),
                            text_color=C["text3"], height=7)
        grip.place(relx=0.5, rely=0.5, anchor="center")

        state = {"y": 0, "h": 0}

        def on_press(e):
            state["y"] = e.y_root
            state["h"] = textbox.winfo_height()

        def on_drag(e):
            delta = e.y_root - state["y"]
            new_h = max(40, state["h"] + delta)
            textbox.configure(height=new_h)

        handle.bind("<Button-1>", on_press)
        handle.bind("<B1-Motion>", on_drag)
        grip.bind("<Button-1>", on_press)
        grip.bind("<B1-Motion>", on_drag)

    def _rewrite_in_place(self, textbox, level):
        """Rewrite text in the textbox using the selected cleanup model."""
        text = textbox.get("1.0", "end").strip()
        if not text:
            return
        prompt = CLEANUP_PROMPTS.get(level)
        if not prompt:
            return
        model = self.cleanup_model_var.get()
        self.update_status(f"Cleanup ({level} · {model})...", "orange")

        def run():
            try:
                result = self._llm_call(prompt, text, model)
                def update():
                    textbox.delete("1.0", "end")
                    textbox.insert("1.0", result)
                    self._clipboard_copy(result)
                    self.update_status("Done — copied", "green")
                self.after(0, update)
            except Exception as e:
                self.after(0, lambda m=str(e)[:60]: self.update_status(f"Error: {m}", "red"))

        threading.Thread(target=run, daemon=True).start()

    def _translate_in_place(self, textbox, lang):
        """Translate text in textbox using DeepSeek."""
        text = textbox.get("1.0", "end").strip()
        if not text:
            return
        self.update_status(f"→ {lang}...", "orange")
        prompt = _prompts.get("translate", _DEFAULT_PROMPTS["translate"]).replace("{lang}", lang)

        def run():
            try:
                result = self._deepseek(prompt, text)
                def update():
                    textbox.delete("1.0", "end")
                    textbox.insert("1.0", result)
                    self._clipboard_copy(result)
                    self.update_status("Translated — copied", "green")
                self.after(0, update)
            except Exception as e:
                self.after(0, lambda m=str(e)[:60]: self.update_status(f"Error: {m}", "red"))

        threading.Thread(target=run, daemon=True).start()

    def _clipboard_copy(self, text):
        pyperclip.copy(text)

    def _copy_from_history(self, text):
        self._clipboard_copy(text)
        self.update_status("Copied", "green")

    def _clear_history(self):
        self.history = []
        self._save_history()
        for w in self.history_frame.winfo_children():
            w.destroy()
        self.update_status("Cleared", "gray")


_lock_fd = None  # kept open for fcntl.flock lifetime

def _check_single_instance():
    """Exit immediately if another instance holds the lock."""
    global _lock_fd
    import atexit

    _lock_fd = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _lock_fd.close()
        raise SystemExit("Another MacDictator instance is running")

    _lock_fd.seek(0)
    _lock_fd.truncate()
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()

    def _cleanup():
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
            os.remove(LOCK_FILE)
        except OSError:
            pass
    atexit.register(_cleanup)


if __name__ == "__main__":
    import signal
    _check_single_instance()
    app = DictatorApp()

    # Signal handlers for tray subprocess communication
    def _on_sigusr1(signum, frame):
        app.after(0, app._toggle_visibility)
    def _on_sigterm(signum, frame):
        app.after(0, app._quit_app)
    signal.signal(signal.SIGUSR1, _on_sigusr1)
    signal.signal(signal.SIGTERM, _on_sigterm)

    app.mainloop()
