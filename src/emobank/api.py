from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import io
import soundfile as sf  # requer python-multipart por causa do upload

from .utils import ensure_artifacts
from .emotion_model import load_emotion
from .risk_model import load_risk

app = FastAPI(title="EmoBank API", version="1.0.0")

# ---------------- Models & Schemas ----------------

class TextInput(BaseModel):
    text: str = Field(..., description="Texto do cliente (chat, transcrição, etc.)")
    amount: float = Field(..., description="Valor da transação")
    channel: str = Field(..., examples=["pix", "card", "chat", "phone"])
    n_prev_tx_7d: int = Field(..., ge=0, description="Qtde de transações anteriores (7 dias)")

def _as_df(payload: TextInput) -> pd.DataFrame:
    return pd.DataFrame([payload.dict()])

@app.on_event("startup")
def load_models():
    base, emo_dir, risk_dir = ensure_artifacts()
    app.state.emo_model, app.state.emo_labels = load_emotion(emo_dir)
    app.state.churn_model, app.state.cols = load_risk(risk_dir, "churned")
    app.state.fraud_model, _ = load_risk(risk_dir, "fraud_flag")

# ---------------- Utility endpoints ----------------

@app.get("/")
def root():
    return {"ok": True, "service": "EmoBank API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ---------------- Main endpoints ----------------

@app.post("/predict_text")
def predict_text(inp: TextInput):
    df = _as_df(inp)

    # emoção prevista vira feature categórica
    emo_idx = app.state.emo_model.predict(df["text"])
    emo = app.state.emo_labels[int(emo_idx[0])]

    # features base
    X = pd.DataFrame({
        "amount": df["amount"],
        "n_prev_tx_7d": df["n_prev_tx_7d"],
        "channel": df["channel"],
        "emotion": [emo],
    })

    # placeholders one-hot p/ todas as colunas vistas no treino
    for col in app.state.cols:
        if col not in X.columns:
            if col.startswith("channel_"):
                X[col] = [1] if ("channel_" + inp.channel) == col else [0]
            elif col.startswith("emotion_"):
                X[col] = [1] if ("emotion_" + emo) == col else [0]
            else:
                X[col] = 0  # numérica contínua ou outra coluna

    # garante mesma ordem/colunas do treino
    X = X.reindex(columns=app.state.cols, fill_value=0)

    churn_p = float(app.state.churn_model.predict_proba(X)[:, 1][0])
    fraud_p = float(app.state.fraud_model.predict_proba(X)[:, 1][0])

    return {
        "emotion": emo,
        "risk": {"churn": round(churn_p, 4), "fraud": round(fraud_p, 4)},
    }

@app.post("/predict_audio")
async def predict_audio(
    file: UploadFile = File(...),
    amount: float = 100.0,
    channel: str = "phone",
    n_prev_tx_7d: int = 3,
):
    # leitura do WAV (mono 16kHz ideal)
    data = await file.read()
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        raise HTTPException(400, detail="Forneça um WAV válido (mono 16kHz de preferência)")

    # STT opcional via faster-whisper
    try:
        from faster_whisper import WhisperModel # type: ignore
        model = WhisperModel("base", device="cpu")
        segments, info = model.transcribe(wav, vad_filter=True)
        text = " ".join([s.text for s in segments])
    except Exception:
        raise HTTPException(
            501,
            detail="STT não habilitado. Instale 'faster-whisper' + 'ffmpeg' ou use /predict_text.",
        )

    return predict_text(TextInput(text=text, amount=amount, channel=channel, n_prev_tx_7d=n_prev_tx_7d))

class ExplainInput(TextInput):
    pass

@app.post("/explain")
def explain(inp: ExplainInput):
    import shap

    df = _as_df(inp)

    # emoção prevista vira feature categórica
    emo_idx = app.state.emo_model.predict(df["text"])
    emo = app.state.emo_labels[int(emo_idx[0])]

    X = pd.DataFrame({
        "amount": df["amount"],
        "n_prev_tx_7d": df["n_prev_tx_7d"],
        "channel": df["channel"],
        "emotion": [emo],
    })

    # placeholders one-hot (mesma lógica do predict_text)
    for col in app.state.cols:
        if col not in X.columns:
            if col.startswith("channel_"):
                X[col] = [1] if ("channel_" + inp.channel) == col else [0]
            elif col.startswith("emotion_"):
                X[col] = [1] if ("emotion_" + emo) == col else [0]
            else:
                X[col] = 0

    X = X.reindex(columns=app.state.cols, fill_value=0)

    # SHAP universal: normaliza formato e evita lista por classe
    explainer = shap.Explainer(app.state.churn_model)
    sv = explainer(X)                 # shap.Explanation
    vals = np.array(sv.values)[0].reshape(-1)  # 1 amostra -> vetor (n_features,)

    # mapeia colunas -> contribuições
    vals = vals[:len(app.state.cols)]
    contrib = dict(zip(app.state.cols, map(float, vals)))

    return {
        "emotion": emo,
        "risk_pred": float(app.state.churn_model.predict_proba(X)[:, 1][0]),
        "shap_contrib": contrib,
    }
