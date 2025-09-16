from pathlib import Path
from src.emobank.data import make_synthetic
from src.emobank.utils import ensure_artifacts

out = Path("data/synthetic.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df = make_synthetic(6000)
df.to_csv(out, index=False)
print(f"[OK] Wrote {out}")
