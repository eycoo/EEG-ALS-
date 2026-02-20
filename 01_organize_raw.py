"""
Script 01: Organize Raw EEG Data
--------------------------------
Fungsi: Menyalin dan me-rename file EDF dari struktur folder berantakan (raw/)
        menjadi struktur flat di dataset_staging/ dengan format penamaan standar.

Input:  raw/ (folder dengan struktur: ALS01/time1/scenario1/EEG.edf, dll.)
Output: dataset_staging/ (file .edf dengan nama: ALS01_S01_Sc01.edf, Normal001_S01_Sc01.edf, dll.)
        metadata.csv (peta semua file dengan info subject, session, scenario, quality)

Format penamaan output:
  - ALS subjects:    ALS01_S01_Sc01.edf   (ALS + 2 digit)
  - Normal subjects: Normal001_S01_Sc01.edf (Normal + 3 digit)
  - S = Session (time), Sc = Scenario

Cara pakai: python 01_organize_raw.py
"""

import os
import re
import shutil
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "raw"
STAGING_DIR = BASE_DIR / "dataset_staging"
METADATA_FILE = BASE_DIR / "metadata.csv"


def extract_info(path_str):
    path_str = str(path_str).replace("\\", "/")
    
    subject = re.search(r"(ALS|id)(\d+)", path_str, re.IGNORECASE)
    session = re.search(r"time(\d+)", path_str, re.IGNORECASE)
    scenario = re.search(r"scenario(\d+)", path_str, re.IGNORECASE)
    
    if not subject or not scenario:
        return None
    
    prefix, num = subject.groups()
    num = int(num)
    
    if prefix.lower() == "als":
        group = "ALS"
        subject_id = f"ALS{num:02d}"
    else:
        group = "Normal"
        subject_id = f"Normal{num:03d}"
    
    session_num = int(session.group(1)) if session else 1
    scenario_num = int(scenario.group(1))
    
    return {
        "subject_id": subject_id,
        "group": group,
        "session": session_num,
        "scenario": scenario_num
    }


def build_filename(info):
    return f"{info['subject_id']}_S{info['session']:02d}_Sc{info['scenario']:02d}.edf"


def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    
    records = []
    copied = 0
    
    for edf_file in RAW_DIR.rglob("EEG.edf"):
        info = extract_info(str(edf_file))
        
        if not info:
            continue
        
        new_name = build_filename(info)
        dest_path = STAGING_DIR / new_name
        
        if not dest_path.exists():
            shutil.copy2(edf_file, dest_path)
            copied += 1
        
        records.append({
            "filename": new_name,
            "subject_id": info["subject_id"],
            "group": info["group"],
            "session": info["session"],
            "scenario": info["scenario"],
            "original_path": str(edf_file.relative_to(BASE_DIR)),
            "quality_flag": "good"
        })
    
    if records:
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["filename"])
        df = df.sort_values(["group", "subject_id", "session", "scenario"])
        df.to_csv(METADATA_FILE, index=False)
    
    print(f"Files copied: {copied}")
    print(f"Staging dir:  {STAGING_DIR}")
    print(f"Metadata:     {METADATA_FILE}")


if __name__ == "__main__":
    main()
