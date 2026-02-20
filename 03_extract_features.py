"""
Script 03: Ekstraksi Fitur EEG per Subband
-------------------------------------------
Fungsi: Membaca file CSV hasil preprocessing, membagi sinyal ke subband frekuensi,
        lalu mengekstrak fitur statistik dan power per subband.

Subband Frekuensi:
  - Delta: 1-4 Hz   (Tidur dalam, proses tak sadar)
  - Theta: 4-8 Hz   (Kantuk, meditasi, memori)
  - Alpha: 8-13 Hz  (Relaksasi, mata tertutup)
  - Beta:  13-30 Hz (Fokus, konsentrasi aktif)
  - Gamma: 30-45 Hz (Pemrosesan kognitif tinggi)

Fitur yang diekstrak per channel per subband:
  - Band Power (uV^2)
  - Relative Power (%)
  - Mean Amplitude
  - Standard Deviation
  - Variance
  - Peak Frequency

Input:  Dataset_CSV_Split/[Task]/*.csv
Output: Reports_Features/
          master_features.csv (gabungan semua fitur)
          features_per_task/Resting_features.csv, Thinking_features.csv, dll.

Cara pakai: python 03_extract_features.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sig
from scipy.fft import fft, fftfreq


BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "Dataset_CSV_Split"
OUTPUT_DIR = BASE_DIR / "Reports_Features"

SFREQ = 500.0

SUBBANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

SELECTED_CHANNELS = None

TASK_FOLDERS = ["Resting", "Thinking", "Typing", "Thinking_Acting"]


def get_eeg_channels(columns):
    exclude = ["time", "timestamp", "index", "sample"]
    channels = [c for c in columns if c.lower() not in exclude]
    
    if SELECTED_CHANNELS:
        channels = [c for c in channels if c in SELECTED_CHANNELS]
    
    return channels


def bandpass_filter(data, low, high, fs):
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq
    
    if low_norm <= 0:
        low_norm = 0.001
    if high_norm >= 1:
        high_norm = 0.999
    
    b, a = sig.butter(4, [low_norm, high_norm], btype="band")
    filtered = sig.filtfilt(b, a, data)
    return filtered


def compute_band_power(data, fs):
    n = len(data)
    freqs = fftfreq(n, 1/fs)
    fft_vals = fft(data)
    psd = np.abs(fft_vals) ** 2 / n
    
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    psd = psd[positive_mask]
    
    return freqs, psd


def extract_features_single_channel(data, fs):
    features = {}
    total_power = 0
    band_powers = {}
    
    for band_name, (low, high) in SUBBANDS.items():
        filtered = bandpass_filter(data, low, high, fs)
        freqs, psd = compute_band_power(filtered, fs)
        
        band_mask = (freqs >= low) & (freqs <= high)
        band_psd = psd[band_mask]
        band_freqs = freqs[band_mask]
        
        power = np.sum(band_psd)
        band_powers[band_name] = power
        total_power += power
        
        features[f"{band_name}_power"] = power
        features[f"{band_name}_mean"] = np.mean(filtered)
        features[f"{band_name}_std"] = np.std(filtered)
        features[f"{band_name}_var"] = np.var(filtered)
        
        if len(band_psd) > 0 and np.sum(band_psd) > 0:
            peak_idx = np.argmax(band_psd)
            features[f"{band_name}_peak_freq"] = band_freqs[peak_idx]
        else:
            features[f"{band_name}_peak_freq"] = (low + high) / 2
    
    if total_power > 0:
        for band_name in SUBBANDS.keys():
            features[f"{band_name}_relative"] = band_powers[band_name] / total_power
    else:
        for band_name in SUBBANDS.keys():
            features[f"{band_name}_relative"] = 0
    
    features["alpha_beta_ratio"] = (
        band_powers["alpha"] / band_powers["beta"] 
        if band_powers["beta"] > 0 else 0
    )
    features["theta_alpha_ratio"] = (
        band_powers["theta"] / band_powers["alpha"] 
        if band_powers["alpha"] > 0 else 0
    )
    features["delta_theta_ratio"] = (
        band_powers["delta"] / band_powers["theta"] 
        if band_powers["theta"] > 0 else 0
    )
    
    return features


def parse_filename(filename):
    parts = filename.replace(".csv", "").split("_")
    
    subject = parts[0]
    session = parts[1]
    scenario = parts[2]
    
    if len(parts) >= 6 and parts[4] == "Acting":
        task = "Thinking_Acting"
        segment = parts[5] if len(parts) > 5 else "01"
    else:
        task = parts[3]
        segment = parts[4] if len(parts) > 4 else "01"
    
    group = "ALS" if subject.startswith("ALS") else "Normal"
    
    return {
        "subject_id": subject,
        "group": group,
        "session": session,
        "scenario": scenario,
        "task": task,
        "segment": segment
    }


def process_single_csv(csv_path):
    df = pd.read_csv(csv_path)
    channels = get_eeg_channels(df.columns)
    
    if not channels:
        return None
    
    file_info = parse_filename(csv_path.name)
    all_features = {}
    all_features.update(file_info)
    all_features["filename"] = csv_path.name
    
    global SFREQ
    if "time" in df.columns:
        times = df["time"].values
        if len(times) > 1:
            dt = np.median(np.diff(times))
            if dt > 0:
                SFREQ = 1.0 / dt
    
    for ch in channels:
        data = df[ch].values
        
        if len(data) < SFREQ:
            continue
        
        ch_features = extract_features_single_channel(data, SFREQ)
        
        for feat_name, feat_val in ch_features.items():
            all_features[f"{ch}_{feat_name}"] = feat_val
    
    return all_features


def aggregate_channel_features(row, channels, feature_type):
    vals = []
    for ch in channels:
        key = f"{ch}_{feature_type}"
        if key in row and pd.notna(row[key]):
            vals.append(row[key])
    return np.mean(vals) if vals else np.nan


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_output_dir = OUTPUT_DIR / "features_per_task"
    task_output_dir.mkdir(exist_ok=True)
    
    all_records = []
    
    for task in TASK_FOLDERS:
        task_dir = INPUT_DIR / task
        if not task_dir.exists():
            continue
        
        csv_files = sorted(task_dir.glob("*.csv"))
        task_records = []
        
        for csv_path in csv_files:
            features = process_single_csv(csv_path)
            if features:
                task_records.append(features)
                all_records.append(features)
        
        if task_records:
            task_df = pd.DataFrame(task_records)
            task_df.to_csv(task_output_dir / f"{task}_features.csv", index=False)
    
    if all_records:
        master_df = pd.DataFrame(all_records)
        master_df.to_csv(OUTPUT_DIR / "master_features.csv", index=False)
        
        summary_cols = [
            "subject_id", "group", "session", "task", "segment"
        ]
        
        channels = get_eeg_channels(pd.read_csv(
            list((INPUT_DIR / TASK_FOLDERS[0]).glob("*.csv"))[0]
        ).columns)
        
        for band in SUBBANDS.keys():
            master_df[f"avg_{band}_power"] = master_df.apply(
                lambda r: aggregate_channel_features(r, channels, f"{band}_power"), 
                axis=1
            )
            master_df[f"avg_{band}_relative"] = master_df.apply(
                lambda r: aggregate_channel_features(r, channels, f"{band}_relative"), 
                axis=1
            )
        
        master_df.to_csv(OUTPUT_DIR / "master_features.csv", index=False)
        
        print(f"Total files processed: {len(all_records)}")
        print(f"Output: {OUTPUT_DIR / 'master_features.csv'}")
        print(f"Task features: {task_output_dir}")


if __name__ == "__main__":
    main()
