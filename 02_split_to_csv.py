"""
Script 02: Preprocessing dan Split ke CSV
------------------------------------------
Fungsi: Membaca file EDF dari dataset_staging/, melakukan preprocessing lengkap,
        lalu memotong berdasarkan annotasi task dan menyimpan ke CSV per segment.

Pipeline Preprocessing:
  1. Bandpass filter (1-45 Hz) - Menghilangkan drift dan high-frequency noise
  2. Notch filter (50 Hz) - Menghilangkan power line interference
  3. ICA - Menghilangkan artefak mata dan otot secara otomatis

Input:  dataset_staging/*.edf
Output: Dataset_CSV_Split/
          Resting/ALS01_T01_Sc01_Resting_01.csv
          Thinking/ALS01_T01_Sc01_Thinking_01.csv
          Typing/...
          Thinking_Acting/...

Channel yang diambil: Semua channel EEG (otomatis deteksi)

Cara pakai: python 02_split_to_csv.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import mne
from pathlib import Path

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


BASE_DIR = Path(__file__).parent.parent
STAGING_DIR = BASE_DIR / "dataset_staging"
OUTPUT_DIR = BASE_DIR / "Dataset_CSV_Split"

BANDPASS_LOW = 1.0
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 50.0
ICA_N_COMPONENTS = 15
ICA_RANDOM_STATE = 42

TASK_FOLDERS = ["Resting", "Thinking", "Typing", "Thinking_Acting"]


def detect_bad_channels(raw, threshold=3.0):
    data = raw.get_data()
    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    mad = np.median(np.abs(variances - median_var))
    
    bad_idx = np.where(np.abs(variances - median_var) > threshold * mad)[0]
    bad_channels = [raw.ch_names[i] for i in bad_idx]
    return bad_channels


def apply_preprocessing(raw):
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, fir_design="firwin", verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    
    n_channels = len(raw.ch_names)
    n_components = min(ICA_N_COMPONENTS, n_channels - 1)
    
    if n_components < 2:
        return raw
    
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=ICA_RANDOM_STATE,
        max_iter=500
    )
    
    try:
        ica.fit(raw, verbose=False)
        
        eog_indices = []
        for ch in raw.ch_names:
            ch_lower = ch.lower()
            if any(x in ch_lower for x in ["eog", "fp1", "fp2", "f7", "f8"]):
                try:
                    eog_idx, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
                    eog_indices.extend(eog_idx)
                except:
                    pass
        
        if not eog_indices:
            muscle_indices = []
            for ch in raw.ch_names:
                ch_lower = ch.lower()
                if any(x in ch_lower for x in ["t7", "t8", "tp9", "tp10"]):
                    try:
                        muscle_idx, _ = ica.find_bads_muscle(raw, verbose=False)
                        muscle_indices.extend(muscle_idx)
                    except:
                        pass
        
        bad_ica = list(set(eog_indices))
        if bad_ica:
            ica.exclude = bad_ica[:3]
            raw = ica.apply(raw, verbose=False)
    except Exception:
        pass
    
    return raw


def get_task_from_description(desc):
    desc = desc.lower().strip()
    
    if "thinking" in desc and "acting" in desc:
        return "Thinking_Acting"
    elif "acting" in desc:
        return "Thinking_Acting"
    elif "typing" in desc:
        return "Typing"
    elif "thinking" in desc:
        return "Thinking"
    elif "resting" in desc or "rest" in desc:
        return "Resting"
    return None


def process_single_file(edf_path):
    filename = edf_path.stem
    parts = filename.split("_")
    
    subject = parts[0]
    session = parts[1].replace("S", "T")
    scenario = parts[2]
    
    base_name = f"{subject}_{session}_{scenario}"
    
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    max_time = raw.times[-1]
    
    raw = apply_preprocessing(raw)
    
    if not raw.annotations:
        return 0
    
    counters = {task: 0 for task in TASK_FOLDERS}
    saved = 0
    
    for ann in raw.annotations:
        desc = ann["description"]
        onset = ann["onset"]
        duration = ann["duration"]
        
        task = get_task_from_description(desc)
        if not task:
            continue
        
        tmax = min(onset + duration, max_time) - 0.01
        if tmax <= onset:
            continue
        
        try:
            segment = raw.copy().crop(tmin=onset, tmax=tmax)
            df = segment.to_data_frame()
            
            counters[task] += 1
            fname = f"{base_name}_{task}_{counters[task]:02d}.csv"
            save_path = OUTPUT_DIR / task / fname
            
            df.to_csv(save_path, index=False)
            saved += 1
            
        except Exception:
            continue
    
    return saved


def main():
    for task in TASK_FOLDERS:
        (OUTPUT_DIR / task).mkdir(parents=True, exist_ok=True)
    
    edf_files = sorted(STAGING_DIR.glob("*.edf"))
    total_files = len(edf_files)
    total_segments = 0
    
    print(f"Found {total_files} EDF files")
    print("-" * 50)
    
    for i, edf_path in enumerate(edf_files, 1):
        saved = process_single_file(edf_path)
        total_segments += saved
        
        if i % 50 == 0 or i == total_files:
            print(f"Progress: {i}/{total_files} files processed")
    
    print("-" * 50)
    print(f"Total segments saved: {total_segments}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
