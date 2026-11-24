"""Preprocess raw PhysioNet multimodal gait CSVs into compact per-trial .npy arrays.

Output: data/processed/<MOD>/S<id>_<speed>.npy, float32, shape (T, C)
  EMG : 16 Hz,  12 ch, log(MAV)                          (trailing zero rows removed)
  FP  : 100 Hz, 16 ch, forces/moments / body weight, COP (anti-aliased decimation 1000->100 Hz)
  IMU : 60 Hz,  joint angles + segment angular velocity + sensor free acceleration
  EEG : 100 Hz, 19 ch, 1-40 Hz band-pass, common-average reference (300->100 Hz)
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
OUT = RAW / "processed"
SPEEDS = ["0.5", "0.75", "1"]
EEG_CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
          "T5", "P3", "P4", "T6", "O1", "O2", "A1", "A2"]  # Pz is the reference
IMU_FILES = ["Joint_Angles_ZXY.csv", "Segment_Angular_Velocity.csv", "Sensor_Free_Acceleration.csv"]

NAME_RE = re.compile(r"S(\d+)_(0\.5|0\.75|1)")


def key_of(name):
    m = NAME_RE.search(name)
    return (int(m.group(1)), m.group(2)) if m else None


def body_weights():
    info = pd.read_csv(RAW / "subject-info.csv")
    info = info[info["Subject"].astype(str).str.match(r"^S\d+$")]   # drop summary rows
    return {int(s[1:]): float(w) for s, w in zip(info["Subject"], info["WEIGHT(kg)"])}


def prep_emg(path):
    df = pd.read_csv(path)
    x = df.iloc[:, 1:13].to_numpy(np.float64)
    t = df.iloc[:, 0].to_numpy(np.float64)
    keep = np.abs(x).sum(1) > 0                      # drop zero-padded rows at the end
    last = np.where(keep)[0].max() + 1
    x, t = x[:last], t[:last]
    grid = np.arange(t[0], t[-1], 1 / 16.0)
    x = np.stack([np.interp(grid, t, x[:, c]) for c in range(12)], 1)
    return np.log(np.clip(x, 1e-7, None)).astype(np.float32)


def prep_fp(path, weight_kg):
    names = [f"{s}:{c}" for s in ("1", "2") for c in
             ("Timestamp", "Sync", "Aux", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy")]
    df = pd.read_csv(path)
    if "1:Fx" not in df.columns:                     # some files (e.g. S2) ship without a header
        df = pd.read_csv(path, header=None, names=names)
    cols = [n for n in names if n.split(":")[1] not in ("Timestamp", "Sync", "Aux")]
    x = df[cols].to_numpy(np.float64, copy=True)
    bw = weight_kg * 9.81
    for s in (0, 8):
        x[:, s:s + 6] /= bw                          # forces & moments normalised to body weight
    x = signal.decimate(x, 10, ftype="fir", axis=0, zero_phase=True)
    return x.astype(np.float32)


def prep_imu(folder):
    parts = []
    for f in IMU_FILES:
        df = pd.read_csv(folder / f)
        parts.append(df.iloc[:, 1:].to_numpy(np.float64))
    n = min(len(p) for p in parts)
    x = np.concatenate([p[:n] for p in parts], 1)
    # keep every column (fixed layout across trials); empty sensor slots are all-zero
    return x.astype(np.float32)


def prep_eeg(path):
    df = pd.read_csv(path, comment="#", low_memory=False)
    x = df[EEG_CH[:18]].to_numpy(np.float64)         # 18 scalp channels (Pz = hardware ref)
    sos = signal.butter(4, [1, 40], btype="band", fs=300, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)
    x = x - x.mean(1, keepdims=True)                 # common average reference
    x = signal.resample_poly(x, 1, 3, axis=0)        # 300 -> 100 Hz
    x = np.clip(x, -150, 150)                        # tame motion artefacts (uV)
    return x.astype(np.float32)


def run(mods):
    bw = body_weights()
    jobs = {
        "EMG": (sorted((RAW / "EMG_Data").glob("EMG_S*.csv")), prep_emg),
        "FP": (sorted((RAW / "FP_Data").glob("FP_S*.csv")), None),
        "EEG": (sorted((RAW / "EEG_csv").glob("S*_raw.csv")), prep_eeg),
        "IMU": (sorted(p for p in (RAW / "IMU_Data").glob("IMU_S*") if p.is_dir()), prep_imu),
    }
    for mod in mods:
        files, fn = jobs[mod]
        (OUT / mod).mkdir(parents=True, exist_ok=True)
        ok = 0
        for f in files:
            k = key_of(f.name)
            if k is None:
                continue
            dst = OUT / mod / f"S{k[0]}_{k[1]}.npy"
            if dst.exists():
                ok += 1
                continue
            try:
                x = prep_fp(f, bw[k[0]]) if mod == "FP" else fn(f)
                if not np.isfinite(x).all():
                    x = np.nan_to_num(x)
                np.save(dst, x)
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {mod} {f.name}: {e}")
        print(f"{mod}: {ok} trials -> {OUT / mod}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", nargs="+", default=["EMG", "FP", "IMU", "EEG"])
    run(ap.parse_args().mods)
