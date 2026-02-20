"""
File Konfigurasi EEG Analysis Pipeline
---------------------------------------
Modifikasi parameter di sini untuk menyesuaikan pipeline tanpa mengubah script utama.
Import file ini dari script lain: from config import *
"""

from pathlib import Path


BASE_DIR = Path(__file__).parent


RAW_DIR = BASE_DIR / "raw"
STAGING_DIR = BASE_DIR / "dataset_staging"
CSV_DIR = BASE_DIR / "Dataset_CSV_Split"
REPORTS_DIR = BASE_DIR / "Reports_Features"
METADATA_FILE = BASE_DIR / "metadata.csv"


BANDPASS_LOW = 1.0
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 50.0
ICA_COMPONENTS = 15
ICA_SEED = 42


SFREQ = 500.0


SUBBANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


SELECTED_CHANNELS = None


TASKS = ["Resting", "Thinking", "Typing", "Thinking_Acting"]


TRANSITIONS = [
    ("Resting", "Thinking"),
    ("Resting", "Typing"),
    ("Thinking", "Thinking_Acting"),
    ("Resting", "Thinking_Acting")
]


ALPHA_LEVEL = 0.05
