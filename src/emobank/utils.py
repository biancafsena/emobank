from __future__ import annotations
import json, joblib
from pathlib import Path

def ensure_artifacts():
    base = Path(__file__).resolve().parents[2] / "artifacts"
    emo_dir = base / "emotion"
    risk_dir = base / "risk"
    base.mkdir(parents=True, exist_ok=True)
    emo_dir.mkdir(parents=True, exist_ok=True)
    risk_dir.mkdir(parents=True, exist_ok=True)
    return base, emo_dir, risk_dir
