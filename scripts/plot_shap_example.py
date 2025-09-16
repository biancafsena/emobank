import json, pandas as pd, matplotlib.pyplot as plt, requests

payload = {
  "text": "Já liguei 3 vezes e ninguém resolve meu problema! Estou muito irritado.",
  "amount": 1200.5, "channel": "pix", "n_prev_tx_7d": 18
}
r = requests.post("http://localhost:8000/explain", json=payload, timeout=15)
r.raise_for_status()
data = r.json()

contrib = pd.Series(data["shap_contrib"]).sort_values(key=lambda s: s.abs(), ascending=True).tail(10)

plt.figure(figsize=(8, 5))
contrib.plot(kind="barh")
plt.title("Top 10 contribuições SHAP – risco de churn (exemplo)")
plt.xlabel("Impacto na previsão (±)")
plt.tight_layout()

import os
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/shap_top.png", dpi=160)
print("OK -> artifacts/shap_top.png")
