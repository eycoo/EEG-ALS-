"""
Script 04: Eksperimen dan Visualisasi
---------------------------------------
Fungsi: Menganalisis perbedaan fitur EEG antara ALS dan Normal,
        menghitung delta transisi antar task per pasien, lalu rata-rata per group.

Metodologi:
  1. Untuk setiap pasien, hitung delta fitur dari Task A -> Task B
  2. Kumpulkan semua delta dari pasien Normal, rata-ratakan
  3. Kumpulkan semua delta dari pasien ALS, rata-ratakan
  4. Bandingkan rata-rata delta Normal vs ALS

Konfigurasi yang bisa diubah:
  - SCENARIOS: Filter scenario tertentu (misal ["Sc01"] atau None untuk semua)
  - SESSIONS: Filter session tertentu (misal ["T01"] atau None untuk semua)
            ALS memiliki session T01-T10, Normal hanya T01
  - CHANNELS: Filter channel tertentu (misal ["C3"] atau None untuk rata-rata semua)
  - TRANSITIONS: Daftar transisi task yang dianalisis
  - EXPERIMENT_REGISTRY: Daftarkan eksperimen custom

Cara pakai: python 04_experiment_plot.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "Reports_Features"
OUTPUT_DIR = BASE_DIR / "Reports_Features"

FEATURES_FILE = INPUT_DIR / "master_features.csv"

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
TASKS = ["Resting", "Thinking", "Typing", "Thinking_Acting"]


# ============================================================================
# KONFIGURASI SCENARIO - Filter scenario tertentu
# Set None untuk menggunakan semua scenario
# Contoh: ["Sc01"] untuk scenario 1 saja, ["Sc01", "Sc02"] untuk beberapa
# ============================================================================
SCENARIOS = ["Sc01"]  # Ubah sesuai kebutuhan, None = semua scenario


# ============================================================================
# KONFIGURASI SESSION - Filter session tertentu
# Set None untuk menggunakan semua session
# ALS memiliki session T01-T10, Normal hanya T01
# Contoh: ["T01"] untuk session 1 saja, ["T01", "T02"] untuk beberapa
# ============================================================================
SESSIONS = ["T01"]  # Ubah sesuai kebutuhan, None = semua session


# ============================================================================
# KONFIGURASI JUMLAH SUBJECT - Batasi jumlah subject yang dianalisis
# Set None untuk menggunakan semua subject
# Contoh: MAX_SUBJECTS_ALS = 5 untuk 5 subject ALS pertama
#         MAX_SUBJECTS_NORMAL = 50 untuk 50 subject Normal pertama
# ============================================================================
MAX_SUBJECTS_ALS = None      # None = semua ALS, atau angka (misal 5)
MAX_SUBJECTS_NORMAL = None   # None = semua Normal, atau angka (misal 50)


# ============================================================================
# KONFIGURASI CHANNEL - Filter channel tertentu
# Set None untuk rata-rata semua channel (menggunakan avg_*_power/relative)
# Contoh: ["C3"] untuk channel C3 saja, ["C3", "C4", "Cz"] untuk beberapa
# ============================================================================
CHANNELS = ["C3"]  # Ubah sesuai kebutuhan, None = rata-rata semua channel


# ============================================================================
# KONFIGURASI TRANSISI - Tambahkan transisi baru di sini
# Format: (task_awal, task_akhir)
# ============================================================================
TRANSITIONS = [
    ("Resting", "Thinking"),
    ("Resting", "Typing"),
    ("Thinking", "Thinking_Acting"),
    ("Resting", "Thinking_Acting"),
]


# ============================================================================
# KONFIGURASI FITUR - Pilih fitur yang akan dianalisis
# Opsi: "relative", "power", "mean", "std", "var", "peak_freq"
# ============================================================================
FEATURE_TYPE = "var"  # Ubah sesuai kebutuhan


# ============================================================================
# KONFIGURASI SUBBAND - Filter subband tertentu
# Set None untuk menggunakan semua band (delta, theta, alpha, beta, gamma)
# Contoh: ["delta"] untuk delta saja, ["alpha", "beta"] untuk alpha dan beta
# ============================================================================
BANDS_TO_ANALYZE = ["alpha"]  # None = semua band, atau list band spesifik


def get_bands_to_analyze():
    """Return bands yang akan dianalisis berdasarkan konfigurasi."""
    if BANDS_TO_ANALYZE is None:
        return BANDS
    return [b for b in BANDS_TO_ANALYZE if b in BANDS]


def load_data():
    df = pd.read_csv(FEATURES_FILE, low_memory=False)
    return df


def filter_by_scenario(df, scenarios=None):
    """Filter dataframe berdasarkan scenario tertentu."""
    if scenarios is None:
        return df
    return df[df["scenario"].isin(scenarios)]


def filter_by_session(df, sessions=None):
    """Filter dataframe berdasarkan session tertentu."""
    if sessions is None:
        return df
    return df[df["session"].isin(sessions)]


def filter_data(df, scenarios=None, sessions=None):
    """Filter dataframe berdasarkan scenario dan session."""
    filtered = filter_by_scenario(df, scenarios)
    filtered = filter_by_session(filtered, sessions)
    return filtered


def limit_subjects(df, max_als=None, max_normal=None):
    """
    Batasi jumlah subject yang dianalisis.
    
    Parameters:
    - max_als: Maksimum subject ALS (None = semua)
    - max_normal: Maksimum subject Normal (None = semua)
    """
    if max_als is None and max_normal is None:
        return df
    
    dfs = []
    
    # Filter ALS
    als_df = df[df["group"] == "ALS"]
    if max_als is not None:
        als_subjects = sorted(als_df["subject_id"].unique())[:max_als]
        als_df = als_df[als_df["subject_id"].isin(als_subjects)]
    dfs.append(als_df)
    
    # Filter Normal
    normal_df = df[df["group"] == "Normal"]
    if max_normal is not None:
        normal_subjects = sorted(normal_df["subject_id"].unique())[:max_normal]
        normal_df = normal_df[normal_df["subject_id"].isin(normal_subjects)]
    dfs.append(normal_df)
    
    return pd.concat(dfs, ignore_index=True)


def get_feature_column(band, channel=None, feature_type="relative"):
    """
    Dapatkan nama kolom fitur berdasarkan band dan channel.
    
    Jika channel=None, gunakan rata-rata (avg_*_relative)
    Jika channel spesifik, gunakan kolom channel tersebut (C3_alpha_relative)
    """
    if channel is None:
        return f"avg_{band}_{feature_type}"
    else:
        return f"{channel}_{band}_{feature_type}"


def get_channel_features(df, band, channels=None, feature_type="relative"):
    """
    Ambil nilai fitur untuk channel tertentu atau rata-rata jika channels=None.
    
    Jika channels berisi beberapa channel, hitung rata-ratanya.
    """
    if channels is None:
        col = f"avg_{band}_{feature_type}"
        if col in df.columns:
            return df[col]
        return None
    
    if len(channels) == 1:
        col = f"{channels[0]}_{band}_{feature_type}"
        if col in df.columns:
            return df[col]
        return None
    
    cols = [f"{ch}_{band}_{feature_type}" for ch in channels]
    existing_cols = [c for c in cols if c in df.columns]
    if existing_cols:
        return df[existing_cols].mean(axis=1)
    return None


def get_avg_features(df):
    power_cols = [col for col in df.columns if col.startswith("avg_") and "_power" in col]
    relative_cols = [col for col in df.columns if col.startswith("avg_") and "_relative" in col]
    
    if not power_cols:
        raw_cols = [col for col in df.columns if "_power" in col and not col.startswith("avg_")]
        
        for band in BANDS:
            band_cols = [col for col in raw_cols if f"{band}_power" in col]
            if band_cols:
                df[f"avg_{band}_power"] = df[band_cols].mean(axis=1)
        
        raw_rel = [col for col in df.columns if "_relative" in col and not col.startswith("avg_")]
        for band in BANDS:
            band_cols = [col for col in raw_rel if f"{band}_relative" in col]
            if band_cols:
                df[f"avg_{band}_relative"] = df[band_cols].mean(axis=1)
    
    return df


# ============================================================================
# FUNGSI INTI: Hitung delta per pasien, lalu rata-rata per group
# ============================================================================
def compute_subject_delta_band(df, subject_id, from_task, to_task, band, channels=None):
    """
    Hitung delta fitur untuk 1 pasien dari task A ke task B untuk 1 band.
    Menggunakan channel spesifik jika ditentukan, atau rata-rata jika channels=None.
    
    Returns: dict dengan from_val, to_val, delta atau None jika data tidak tersedia
    """
    subj_df = df[df["subject_id"] == subject_id]
    
    from_df = subj_df[subj_df["task"] == from_task]
    to_df = subj_df[subj_df["task"] == to_task]
    
    from_vals = get_channel_features(from_df, band, channels, FEATURE_TYPE)
    to_vals = get_channel_features(to_df, band, channels, FEATURE_TYPE)
    
    if from_vals is None or to_vals is None:
        return None
    
    from_vals = from_vals.dropna()
    to_vals = to_vals.dropna()
    
    if len(from_vals) == 0 or len(to_vals) == 0:
        return None
    
    from_mean = from_vals.mean()
    to_mean = to_vals.mean()
    
    return {
        "from_val": from_mean,
        "to_val": to_mean,
        "delta": to_mean - from_mean
    }


def compute_group_transition_deltas(df, from_task, to_task, band, channels=None, scenarios=None, sessions=None):
    """
    Hitung delta transisi per pasien, lalu kumpulkan per group.
    
    Proses:
    1. Filter berdasarkan scenario dan session jika ditentukan
    2. Untuk setiap pasien dalam group, hitung: delta = mean(to_task) - mean(from_task)
    3. Kumpulkan semua delta per group
    4. Hitung rata-rata delta untuk masing-masing group
    """
    filtered_df = filter_data(df, scenarios, sessions)
    results = {}
    
    for group in ["ALS", "Normal"]:
        group_df = filtered_df[filtered_df["group"] == group]
        subjects = group_df["subject_id"].unique()
        
        subject_deltas = []
        for subj in subjects:
            result = compute_subject_delta_band(group_df, subj, from_task, to_task, band, channels)
            if result is not None:
                subject_deltas.append({
                    "subject_id": subj,
                    "from_val": result["from_val"],
                    "to_val": result["to_val"],
                    "delta": result["delta"]
                })
        
        if subject_deltas:
            deltas = [d["delta"] for d in subject_deltas]
            from_vals = [d["from_val"] for d in subject_deltas]
            to_vals = [d["to_val"] for d in subject_deltas]
            results[group] = {
                "subject_deltas": subject_deltas,
                "mean_from": np.mean(from_vals),
                "mean_to": np.mean(to_vals),
                "mean_delta": np.mean(deltas),
                "std_delta": np.std(deltas),
                "sem_delta": np.std(deltas) / np.sqrt(len(deltas)),
                "n_subjects": len(deltas),
                "all_deltas": deltas
            }
    
    return results


def compare_group_deltas(als_deltas, normal_deltas):
    """Bandingkan delta ALS vs Normal dengan uji statistik."""
    if len(als_deltas) < 2 or len(normal_deltas) < 2:
        return None
    
    t_stat, t_pval = stats.ttest_ind(als_deltas, normal_deltas)
    u_stat, u_pval = stats.mannwhitneyu(als_deltas, normal_deltas, alternative="two-sided")
    
    pooled_std = np.sqrt((np.std(als_deltas)**2 + np.std(normal_deltas)**2) / 2)
    cohens_d = (np.mean(als_deltas) - np.mean(normal_deltas)) / pooled_std if pooled_std > 0 else 0
    
    return {
        "t_statistic": t_stat,
        "t_pvalue": t_pval,
        "u_statistic": u_stat,
        "u_pvalue": u_pval,
        "cohens_d": cohens_d,
        "significant_05": t_pval < 0.05,
        "significant_01": t_pval < 0.01
    }


# ============================================================================
# CUSTOM EXPERIMENTS - Tambahkan eksperimen baru di sini
# ============================================================================

def experiment_transition_delta(df, from_task, to_task, bands=None, channels=None, scenarios=None, sessions=None):
    """
    Eksperimen standar: bandingkan delta transisi ALS vs Normal.
    
    Untuk setiap band:
    - Hitung delta per pasien (TaskB - TaskA)
    - Rata-ratakan delta untuk ALS dan Normal
    - Bandingkan dengan t-test
    
    Parameters:
    - channels: list channel spesifik (None = gunakan rata-rata semua channel)
    - scenarios: list scenario untuk filter (None = semua scenario)
    - sessions: list session untuk filter (None = semua session)
    """
    if bands is None:
        bands = get_bands_to_analyze()
    if channels is None:
        channels = CHANNELS if CHANNELS else None
    if scenarios is None:
        scenarios = SCENARIOS if SCENARIOS else None
    if sessions is None:
        sessions = SESSIONS if SESSIONS else None
    
    channel_str = ", ".join(channels) if channels else "all"
    scenario_str = ", ".join(scenarios) if scenarios else "all"
    session_str = ", ".join(sessions) if sessions else "all"
    
    results = []
    for band in bands:
        group_deltas = compute_group_transition_deltas(df, from_task, to_task, band, channels, scenarios, sessions)
        
        record = {
            "transition": f"{from_task} -> {to_task}",
            "from_task": from_task,
            "to_task": to_task,
            "band": band,
            "feature": FEATURE_TYPE,
            "channels": channel_str,
            "scenarios": scenario_str,
            "sessions": session_str
        }
        
        if "ALS" in group_deltas:
            record["als_from_mean"] = group_deltas["ALS"]["mean_from"]
            record["als_to_mean"] = group_deltas["ALS"]["mean_to"]
            record["als_mean_delta"] = group_deltas["ALS"]["mean_delta"]
            record["als_std_delta"] = group_deltas["ALS"]["std_delta"]
            record["als_n"] = group_deltas["ALS"]["n_subjects"]
        
        if "Normal" in group_deltas:
            record["normal_from_mean"] = group_deltas["Normal"]["mean_from"]
            record["normal_to_mean"] = group_deltas["Normal"]["mean_to"]
            record["normal_mean_delta"] = group_deltas["Normal"]["mean_delta"]
            record["normal_std_delta"] = group_deltas["Normal"]["std_delta"]
            record["normal_n"] = group_deltas["Normal"]["n_subjects"]
        
        if "ALS" in group_deltas and "Normal" in group_deltas:
            comparison = compare_group_deltas(
                group_deltas["ALS"]["all_deltas"],
                group_deltas["Normal"]["all_deltas"]
            )
            if comparison:
                record.update(comparison)
        
        results.append(record)
    
    return pd.DataFrame(results)


def experiment_placeholder_1(df):
    """
    PLACEHOLDER: Tambahkan eksperimen custom Anda di sini.
    
    Contoh: Bandingkan rasio alpha/beta antara task.
    
    Return: DataFrame dengan hasil eksperimen
    """
    pass


def experiment_placeholder_2(df):
    """
    PLACEHOLDER: Tambahkan eksperimen custom Anda di sini.
    
    Contoh: Analisis temporal pattern dalam satu task.
    
    Return: DataFrame dengan hasil eksperimen
    """
    pass


# ============================================================================
# REGISTRY EKSPERIMEN - Daftarkan eksperimen yang ingin dijalankan
# Tambahkan nama fungsi baru di sini untuk dieksekusi otomatis
# ============================================================================
EXPERIMENT_REGISTRY = {
    # "nama_eksperimen": fungsi_eksperimen,
    # Contoh: "alpha_beta_ratio": experiment_alpha_beta_ratio,
}


def run_custom_experiments(df):
    """Jalankan semua eksperimen yang terdaftar di EXPERIMENT_REGISTRY."""
    results = {}
    for name, func in EXPERIMENT_REGISTRY.items():
        if func is not None:
            print(f"Running custom experiment: {name}...")
            results[name] = func(df)
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_baseline_comparison(df, save_path):
    """Plot baseline comparison dengan filter scenario, session, dan channel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter by scenario and session
    filtered_df = filter_data(df, SCENARIOS, SESSIONS)
    resting_df = filtered_df[filtered_df["task"] == "Resting"]
    
    channel_str = ", ".join(CHANNELS) if CHANNELS else "all"
    scenario_str = ", ".join(SCENARIOS) if SCENARIOS else "all"
    session_str = ", ".join(SESSIONS) if SESSIONS else "all"
    
    als_powers = []
    normal_powers = []
    
    bands_list = get_bands_to_analyze()
    for band in bands_list:
        als_vals = get_channel_features(resting_df[resting_df["group"] == "ALS"], band, CHANNELS, "power")
        normal_vals = get_channel_features(resting_df[resting_df["group"] == "Normal"], band, CHANNELS, "power")
        als_powers.append(als_vals.mean() if als_vals is not None else 0)
        normal_powers.append(normal_vals.mean() if normal_vals is not None else 0)
    
    x = np.arange(len(bands_list))
    width = 0.35
    
    axes[0].bar(x - width/2, als_powers, width, label="ALS", color="#e74c3c", alpha=0.8)
    axes[0].bar(x + width/2, normal_powers, width, label="Normal", color="#3498db", alpha=0.8)
    axes[0].set_xlabel("Frequency Band")
    axes[0].set_ylabel("Band Power (uV^2)")
    axes[0].set_title(f"Baseline Resting: Band Power\n(Channel: {channel_str}, Scenario: {scenario_str})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([b.capitalize() for b in bands_list])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    
    als_rel = []
    normal_rel = []
    
    for band in bands_list:
        als_vals = get_channel_features(resting_df[resting_df["group"] == "ALS"], band, CHANNELS, FEATURE_TYPE)
        normal_vals = get_channel_features(resting_df[resting_df["group"] == "Normal"], band, CHANNELS, FEATURE_TYPE)
        als_rel.append((als_vals.mean() * 100) if als_vals is not None else 0)
        normal_rel.append((normal_vals.mean() * 100) if normal_vals is not None else 0)
    
    axes[1].bar(x - width/2, als_rel, width, label="ALS", color="#e74c3c", alpha=0.8)
    axes[1].bar(x + width/2, normal_rel, width, label="Normal", color="#3498db", alpha=0.8)
    axes[1].set_xlabel("Frequency Band")
    axes[1].set_ylabel("Relative Power")
    axes[1].set_title(f"Baseline Resting: Relative Power\n(Channel: {channel_str}, Scenario: {scenario_str})")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([b.capitalize() for b in bands_list])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_transition_deltas(df, save_path):
    """
    Plot delta transisi per pasien yang sudah di-rata-ratakan per group.
    Setiap bar menunjukkan rata-rata delta dari semua pasien dalam group.
    Error bar menunjukkan SEM (Standard Error of Mean).
    
    Menggunakan CHANNELS, SCENARIOS, dan SESSIONS dari konfigurasi.
    """
    n_transitions = len(TRANSITIONS)
    n_cols = 2
    n_rows = (n_transitions + 1) // 2
    
    channel_str = ", ".join(CHANNELS) if CHANNELS else "all"
    scenario_str = ", ".join(SCENARIOS) if SCENARIOS else "all"
    session_str = ", ".join(SESSIONS) if SESSIONS else "all"
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_transitions > 1 else [axes]
    
    for idx, (from_task, to_task) in enumerate(TRANSITIONS):
        ax = axes[idx]
        
        als_deltas = []
        normal_deltas = []
        als_sems = []
        normal_sems = []
        
        for band in get_bands_to_analyze():
            group_results = compute_group_transition_deltas(df, from_task, to_task, band, CHANNELS, SCENARIOS, SESSIONS)
            
            als_delta = 0
            normal_delta = 0
            als_sem = 0
            normal_sem = 0
            
            if "ALS" in group_results:
                als_delta = group_results["ALS"]["mean_delta"] * 100
                als_sem = group_results["ALS"]["sem_delta"] * 100
            
            if "Normal" in group_results:
                normal_delta = group_results["Normal"]["mean_delta"] * 100
                normal_sem = group_results["Normal"]["sem_delta"] * 100
            
            als_deltas.append(als_delta)
            normal_deltas.append(normal_delta)
            als_sems.append(als_sem)
            normal_sems.append(normal_sem)
        
        bands_list = get_bands_to_analyze()
        x = np.arange(len(bands_list))
        width = 0.35
        
        ax.bar(x - width/2, als_deltas, width, yerr=als_sems, 
               label="ALS", color="#e74c3c", alpha=0.8, capsize=3)
        ax.bar(x + width/2, normal_deltas, width, yerr=normal_sems,
               label="Normal", color="#3498db", alpha=0.8, capsize=3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel(f"Delta {FEATURE_TYPE.capitalize()}")
        ax.set_title(f"{from_task} -> {to_task}\n(Ch: {channel_str}, Sc: {scenario_str}, Sess: {session_str}, Feat: {FEATURE_TYPE})")
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in bands_list])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    
    for idx in range(len(TRANSITIONS), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_task_comparison(df, save_path):
    """Plot task comparison dengan filter scenario, session, dan channel."""
    # Filter by scenario and session
    filtered_df = filter_data(df, SCENARIOS, SESSIONS)
    
    channel_str = ", ".join(CHANNELS) if CHANNELS else "all"
    scenario_str = ", ".join(SCENARIOS) if SCENARIOS else "all"
    session_str = ", ".join(SESSIONS) if SESSIONS else "all"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Task Comparison (Ch: {channel_str}, Sc: {scenario_str}, Sess: {session_str})", fontsize=12, y=1.02)
    axes = axes.flatten()
    
    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        task_df = filtered_df[filtered_df["task"] == task]
        
        als_vals = []
        normal_vals = []
        als_errs = []
        normal_errs = []
        
        for band in get_bands_to_analyze():
            als_data = get_channel_features(task_df[task_df["group"] == "ALS"], band, CHANNELS, FEATURE_TYPE)
            normal_data = get_channel_features(task_df[task_df["group"] == "Normal"], band, CHANNELS, FEATURE_TYPE)
            
            if als_data is not None:
                als_vals.append(als_data.mean() * 100)
                als_errs.append(als_data.std() * 100 / np.sqrt(len(als_data)) if len(als_data) > 0 else 0)
            else:
                als_vals.append(0)
                als_errs.append(0)
            
            if normal_data is not None:
                normal_vals.append(normal_data.mean() * 100)
                normal_errs.append(normal_data.std() * 100 / np.sqrt(len(normal_data)) if len(normal_data) > 0 else 0)
            else:
                normal_vals.append(0)
                normal_errs.append(0)
        
        bands_list = get_bands_to_analyze()
        x = np.arange(len(bands_list))
        width = 0.35
        
        ax.bar(x - width/2, als_vals, width, yerr=als_errs, label="ALS", 
               color="#e74c3c", alpha=0.8, capsize=3)
        ax.bar(x + width/2, normal_vals, width, yerr=normal_errs, label="Normal", 
               color="#3498db", alpha=0.8, capsize=3)
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("Relative Power")
        ax.set_title(f"Task: {task}")
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in bands_list])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_subject_deltas(df, save_path, csv_path=None):
    """
    Plot delta per subject untuk melihat distribusi positif/negatif.
    
    X-axis: Subject ID
    Y-axis: Delta value (variance atau feature lain)
    Warna: Merah=ALS, Biru=Normal
    Garis horizontal di y=0 memisahkan positif dan negatif
    
    Juga menyimpan CSV dengan detail per subject.
    """
    filtered_df = filter_data(df, SCENARIOS, SESSIONS)
    
    channel_str = ", ".join(CHANNELS) if CHANNELS else "all"
    scenario_str = ", ".join(SCENARIOS) if SCENARIOS else "all"
    session_str = ", ".join(SESSIONS) if SESSIONS else "all"
    
    # Untuk menyimpan ke CSV
    csv_records = []
    
    n_transitions = len(TRANSITIONS)
    fig, axes = plt.subplots(n_transitions, 1, figsize=(16, 4 * n_transitions))
    if n_transitions == 1:
        axes = [axes]
    
    for idx, (from_task, to_task) in enumerate(TRANSITIONS):
        ax = axes[idx]
        
        # Pilih band untuk diplot (gunakan band pertama dari konfigurasi)
        bands_list = get_bands_to_analyze()
        band = bands_list[0]  # band pertama dari konfigurasi
        
        # Kumpulkan delta per subject
        all_subjects = []
        all_deltas = []
        all_colors = []
        all_groups = []
        
        for group in ["ALS", "Normal"]:
            group_df = filtered_df[filtered_df["group"] == group]
            subjects = sorted(group_df["subject_id"].unique())
            
            for subj in subjects:
                result = compute_subject_delta_band(group_df, subj, from_task, to_task, band, CHANNELS)
                if result is not None:
                    all_subjects.append(subj)
                    all_deltas.append(result["delta"])
                    all_colors.append("#e74c3c" if group == "ALS" else "#3498db")
                    all_groups.append(group)
                    
                    # Simpan ke CSV records
                    csv_records.append({
                        "subject_id": subj,
                        "group": group,
                        "band": band,
                        "channel": channel_str,
                        "scenarios": scenario_str,
                        "sessions": session_str,
                        "feature": FEATURE_TYPE,
                        "task_1": from_task,
                        "task_2": to_task,
                        "task_1_value": result["from_val"],
                        "task_2_value": result["to_val"],
                        "delta": result["delta"]
                    })
        
        if not all_deltas:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            continue
        
        # Sort by group (ALS first) then by subject
        combined = list(zip(all_groups, all_subjects, all_deltas, all_colors))
        combined.sort(key=lambda x: (0 if x[0] == "ALS" else 1, x[1]))
        all_groups, all_subjects, all_deltas, all_colors = zip(*combined)
        
        x = np.arange(len(all_subjects))
        
        # Plot bars dengan warna berbeda untuk positif/negatif
        bars = ax.bar(x, all_deltas, color=all_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        
        # Garis horizontal di y=0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        
        # Hitung statistik positif/negatif per group
        als_deltas = [d for g, d in zip(all_groups, all_deltas) if g == "ALS"]
        normal_deltas = [d for g, d in zip(all_groups, all_deltas) if g == "Normal"]
        
        als_pos = sum(1 for d in als_deltas if d > 0)
        als_neg = sum(1 for d in als_deltas if d < 0)
        normal_pos = sum(1 for d in normal_deltas if d > 0)
        normal_neg = sum(1 for d in normal_deltas if d < 0)
        
        # Tambah garis vertikal pemisah ALS dan Normal
        als_count = sum(1 for g in all_groups if g == "ALS")
        if als_count > 0 and als_count < len(all_groups):
            ax.axvline(x=als_count - 0.5, color="gray", linestyle="--", linewidth=1.5)
        
        ax.set_xlabel("Subject ID")
        ax.set_ylabel(f"Delta {FEATURE_TYPE.capitalize()} ({band.capitalize()} band)")
        ax.set_title(f"{from_task} -> {to_task}\n"
                     f"ALS: +{als_pos}/-{als_neg} | Normal: +{normal_pos}/-{normal_neg}\n"
                     f"(Ch: {channel_str}, Sc: {scenario_str}, Sess: {session_str})")
        ax.set_xticks(x)
        ax.set_xticklabels(all_subjects, rotation=90, fontsize=6)
        ax.grid(axis="y", alpha=0.3)
        
        # Legend custom
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", edgecolor="black", label="ALS"),
            Patch(facecolor="#3498db", edgecolor="black", label="Normal")
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Simpan CSV jika path diberikan
    if csv_path and csv_records:
        csv_df = pd.DataFrame(csv_records)
        csv_df.to_csv(csv_path, index=False)


def generate_statistics_report(df, save_path):
    """Generate comparison report for each task dengan filter scenario/session/channel."""
    # Filter by scenario and session
    filtered_df = filter_data(df, SCENARIOS, SESSIONS)
    
    channel_str = ", ".join(CHANNELS) if CHANNELS else "all"
    scenario_str = ", ".join(SCENARIOS) if SCENARIOS else "all"
    session_str = ", ".join(SESSIONS) if SESSIONS else "all"
    
    records = []
    
    for task in TASKS:
        for band in get_bands_to_analyze():
            task_df = filtered_df[filtered_df["task"] == task]
            
            als_data = get_channel_features(task_df[task_df["group"] == "ALS"], band, CHANNELS, FEATURE_TYPE)
            normal_data = get_channel_features(task_df[task_df["group"] == "Normal"], band, CHANNELS, FEATURE_TYPE)
            
            if als_data is None or normal_data is None:
                continue
            
            als_data = als_data.dropna()
            normal_data = normal_data.dropna()
            
            if len(als_data) < 2 or len(normal_data) < 2:
                continue
            
            t_stat, t_pval = stats.ttest_ind(als_data, normal_data)
            u_stat, u_pval = stats.mannwhitneyu(als_data, normal_data, alternative="two-sided")
            
            pooled_std = np.sqrt((als_data.std()**2 + normal_data.std()**2) / 2)
            cohens_d = (als_data.mean() - normal_data.mean()) / pooled_std if pooled_std > 0 else 0
            
            records.append({
                "task": task,
                "band": band,
                "channels": channel_str,
                "scenarios": scenario_str,
                "sessions": session_str,
                "feature": "relative_power",
                "als_mean": als_data.mean(),
                "als_std": als_data.std(),
                "als_n": len(als_data),
                "normal_mean": normal_data.mean(),
                "normal_std": normal_data.std(),
                "normal_n": len(normal_data),
                "t_statistic": t_stat,
                "t_pvalue": t_pval,
                "u_pvalue": u_pval,
                "cohens_d": cohens_d,
                "significant_05": t_pval < 0.05,
                "significant_01": t_pval < 0.01
            })
    
    stats_df = pd.DataFrame(records)
    stats_df.to_csv(save_path, index=False)
    return stats_df


def generate_transition_report(df, save_path):
    """
    Generate transition report dengan metodologi:
    1. Hitung delta per pasien
    2. Rata-ratakan delta per group
    3. Bandingkan ALS vs Normal
    """
    all_records = []
    
    for from_task, to_task in TRANSITIONS:
        transition_df = experiment_transition_delta(df, from_task, to_task)
        all_records.append(transition_df)
    
    trans_df = pd.concat(all_records, ignore_index=True)
    trans_df.to_csv(save_path, index=False)
    return trans_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    df = get_avg_features(df)
    
    # Batasi jumlah subject jika dikonfigurasi
    df = limit_subjects(df, MAX_SUBJECTS_ALS, MAX_SUBJECTS_NORMAL)
    
    channel_str = ", ".join(CHANNELS) if CHANNELS else "all"
    scenario_str = ", ".join(SCENARIOS) if SCENARIOS else "all"
    session_str = ", ".join(SESSIONS) if SESSIONS else "all"
    max_als_str = str(MAX_SUBJECTS_ALS) if MAX_SUBJECTS_ALS else "all"
    max_normal_str = str(MAX_SUBJECTS_NORMAL) if MAX_SUBJECTS_NORMAL else "all"
    
    print("=" * 60)
    print("EEG ALS vs Normal Analysis")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"ALS subjects: {df[df['group'] == 'ALS']['subject_id'].nunique()}")
    print(f"Normal subjects: {df[df['group'] == 'Normal']['subject_id'].nunique()}")
    print(f"Tasks: {df['task'].unique().tolist()}")
    print("-" * 60)
    print("FILTERING CONFIGURATION:")
    print(f"  Channels: {channel_str}")
    print(f"  Scenarios: {scenario_str}")
    print(f"  Sessions: {session_str}")
    bands_str = ", ".join(get_bands_to_analyze()) if BANDS_TO_ANALYZE else "all"
    print(f"  Feature: {FEATURE_TYPE}")
    print(f"  Bands: {bands_str}")
    print(f"  Max ALS subjects: {max_als_str}")
    print(f"  Max Normal subjects: {max_normal_str}")
    print("=" * 60)
    
    # print("\nGenerating baseline comparison plot...")
    # plot_baseline_comparison(df, OUTPUT_DIR / "baseline_comparison.png")
    
    print("\nGenerating transition delta plot...")
    plot_transition_deltas(df, OUTPUT_DIR / "transition_deltas.png")
    
    print("Generating subject delta plot...")
    plot_subject_deltas(df, OUTPUT_DIR / "subject_deltas.png", OUTPUT_DIR / "subject_deltas.csv")
    
    # print("Generating task comparison plot...")
    # plot_task_comparison(df, OUTPUT_DIR / "task_comparison.png")
    
    print("Generating statistics report...")
    stats_df = generate_statistics_report(df, OUTPUT_DIR / "statistics_report.csv")
    
    print("Generating transition report...")
    trans_df = generate_transition_report(df, OUTPUT_DIR / "transition_report.csv")
    
    custom_results = run_custom_experiments(df)
    
    with pd.ExcelWriter(OUTPUT_DIR / "analysis_summary.xlsx") as writer:
        stats_df.to_excel(writer, sheet_name="Group_Comparison", index=False)
        trans_df.to_excel(writer, sheet_name="Transition_Analysis", index=False)
        
        for name, result_df in custom_results.items():
            if result_df is not None:
                sheet_name = name[:31]
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        summary_data = {
            "Metric": [
                "Total Files", 
                "ALS Subjects", 
                "Normal Subjects", 
                "Tasks Analyzed",
                "Transitions Analyzed",
                "Channels Filter",
                "Scenarios Filter",
                "Sessions Filter",
                "Feature Type",
                "Max ALS Subjects",
                "Max Normal Subjects"
            ],
            "Value": [
                len(df),
                df[df["group"] == "ALS"]["subject_id"].nunique(),
                df[df["group"] == "Normal"]["subject_id"].nunique(),
                len(TASKS),
                len(TRANSITIONS),
                channel_str,
                scenario_str,
                session_str,
                FEATURE_TYPE,
                max_als_str,
                max_normal_str
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
    
    print("-" * 60)
    print("Analysis complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Files generated:")
    # print("  - baseline_comparison.png")
    print("  - transition_deltas.png")
    print("  - subject_deltas.png")
    print("  - subject_deltas.csv")
    # print("  - task_comparison.png")
    print("  - statistics_report.csv")
    print("  - transition_report.csv")
    print("  - analysis_summary.xlsx")
    
    if custom_results:
        print("Custom experiments:")
        for name in custom_results.keys():
            print(f"  - {name}")

if __name__ == "__main__":
    main()
