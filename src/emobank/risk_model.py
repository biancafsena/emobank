from __future__ import annotations
import json, joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from .emotion_model import load_emotion

CATEG = ["channel"]
NUM = ["amount","n_prev_tx_7d"]
TARGETS = ["churned","fraud_flag"]

def _build_features(df: pd.DataFrame, emotion_model, labels):
    # Predição de emoção como feature
    emo_idx = emotion_model.predict(df["text"]
        if hasattr(df, "__getitem__") else pd.Series([df]))
    # Se uma única amostra:
    if not isinstance(emo_idx, (list, tuple)) and getattr(emo_idx, "shape", None) == ():
        emo_idx = [int(emo_idx)]
    emo = [labels[i] for i in emo_idx]
    X = pd.DataFrame({
        "amount": df["amount"],
        "n_prev_tx_7d": df["n_prev_tx_7d"],
        "channel": df["channel"],
        "emotion": emo
    })
    X = pd.get_dummies(X, columns=["channel","emotion"], drop_first=True, dtype=int)
    return X

def train_risk(df_train: pd.DataFrame, df_test: pd.DataFrame, emo_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    emotion_model, labels = load_emotion(emo_dir)

    Xtr = _build_features(df_train, emotion_model, labels)
    Xte = _build_features(df_test, emotion_model, labels)

    results = {}
    for target in TARGETS:
        ytr, yte = df_train[target].values, df_test[target].values
        clf = RandomForestClassifier(n_estimators=400, random_state=2025, n_jobs=-1)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:,1]
        pred = (proba>=0.5).astype(int)
        metrics = {
            "accuracy": float((pred==yte).mean()),
            "roc_auc": float(roc_auc_score(yte, proba)),
            "precision_recall_f1": tuple(map(float, precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)[:3])),
            "n": int(len(yte))
        }
        joblib.dump(clf, out_dir / f"{target}_model.joblib")
        (out_dir / f"{target}_metrics.json").write_text(json.dumps(metrics, indent=2))
        results[target] = metrics

    (out_dir / "feature_columns.json").write_text(json.dumps({"columns": list(Xtr.columns)}))

def load_risk(out_dir: Path, target: str):
    import joblib, json
    model = joblib.load(out_dir / f"{target}_model.joblib")
    columns = json.loads((out_dir / "feature_columns.json").read_text())["columns"]
    return model, columns
