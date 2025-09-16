import pandas as pd
from pathlib import Path
from src.emobank.data import split_train_test
from src.emobank.risk_model import train_risk
from src.emobank.utils import ensure_artifacts

df = pd.read_csv("data/synthetic.csv")
tr, te = split_train_test(df, test_size=0.2)
base, emo_dir, risk_dir = ensure_artifacts()
train_risk(tr, te, emo_dir, risk_dir)
print("[OK] Risk models trained ->", risk_dir)
