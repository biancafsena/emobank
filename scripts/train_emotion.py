import pandas as pd
from pathlib import Path
from src.emobank.data import split_train_test
from src.emobank.emotion_model import train_emotion
from src.emobank.utils import ensure_artifacts

df = pd.read_csv("data/synthetic.csv")
tr, te = split_train_test(df, test_size=0.2)
base, emo_dir, _ = ensure_artifacts()
train_emotion(tr, te, emo_dir)
print("[OK] Emotion model trained ->", emo_dir)
