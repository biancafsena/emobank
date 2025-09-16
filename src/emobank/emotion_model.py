from __future__ import annotations
import json, joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from .nlp import make_vectorizer
from .data import EMOTIONS

def train_emotion(df_train: pd.DataFrame, df_test: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    le = LabelEncoder().fit(EMOTIONS)
    y_train = le.transform(df_train["emotion"])
    y_test = le.transform(df_test["emotion"])

    pipe = Pipeline([
        ("tfidf", make_vectorizer(6000)),
        ("clf", LogisticRegression(max_iter=400, n_jobs=None, multi_class="auto"))
    ])
    pipe.fit(df_train["text"], y_train)
    pred = pipe.predict(df_test["text"])
    acc = accuracy_score(y_test, pred)
    rep = classification_report(y_test, pred, target_names=le.classes_, output_dict=True)

    joblib.dump(pipe, out_dir / "emotion_model.joblib")
    (out_dir / "emotion_labels.json").write_text(json.dumps({"classes": list(le.classes_)}))
    (out_dir / "emotion_metrics.json").write_text(json.dumps({"accuracy": acc, "report": rep}, indent=2))

def load_emotion(out_dir: Path):
    import joblib, json
    model = joblib.load(out_dir / "emotion_model.joblib")
    labels = json.loads((out_dir / "emotion_labels.json").read_text())["classes"]
    return model, labels
