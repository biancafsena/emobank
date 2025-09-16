from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

RNG_SEED = 2025
EMOTIONS = ["joy","anger","sadness","fear","neutral"]
CHANNELS = ["pix","card","chat","phone"]

@dataclass
class TransactionRecord:
    text: str
    emotion: str
    amount: float
    channel: str
    n_prev_tx_7d: int
    churned: int  # rótulo para churn
    fraud_flag: int  # rótulo para fraude emocional (heurístico)

def _sample_text(emotion: str, rng: np.random.Generator) -> str:
    samples = {
        "joy": [
            "Experiência excelente, muito obrigado pela rapidez!",
            "Atendimento perfeito, estou satisfeita com o serviço.",
            "Fiquei feliz com a aprovação imediata."
        ],
        "anger": [
            "Estou cansada de ligar e ninguém resolver meu problema!",
            "Péssimo suporte, descaso total com o cliente.",
            "Isso é um absurdo, cancelem tudo agora."
        ],
        "sadness": [
            "Estou desapontada, esperava mais consideração.",
            "Que pena, tive uma experiência ruim desta vez.",
            "Me sinto desamparada pelo suporte."
        ],
        "fear": [
            "Estou preocupada com uma movimentação estranha na conta.",
            "Tenho medo de ter sido vítima de fraude.",
            "Algo não parece seguro nessa transação."
        ],
        "neutral": [
            "Solicito segunda via do boleto.",
            "Qual é o status da minha solicitação?",
            "Preciso alterar meu endereço cadastral."
        ]
    }
    return rng.choice(samples[emotion])

def make_synthetic(n: int = 5000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        emotion = rng.choice(EMOTIONS, p=[0.22,0.18,0.18,0.12,0.30])
        text = _sample_text(emotion, rng)
        channel = rng.choice(CHANNELS, p=[0.4,0.35,0.15,0.10])
        amount = float(np.round(rng.gamma(2.2, 200),2))
        n_prev = int(np.clip(rng.poisson(5),0,50))

        # Heurísticas de rótulo secundário
        churn_risk = 0.0
        churn_risk += 0.9 if emotion in ["anger","sadness","fear"] else -0.3
        churn_risk += 0.4 if channel in ["chat","phone"] else 0.0
        churn_risk += 0.002 * (amount - 200)  # tickets mais altos
        churn_risk += 0.05 * max(0, 10 - n_prev)  # baixa frequência + risco
        p_churn = 1/(1+np.exp(-churn_risk))
        churned = int(rng.random() < p_churn*0.35)

        fraud_risk = 0.0
        fraud_risk += 1.0 if emotion in ["fear","anger"] else -0.4
        fraud_risk += 0.7 if channel == "pix" else 0.0
        fraud_risk += 0.003 * (amount - 300)
        p_fraud = 1/(1+np.exp(-fraud_risk))
        fraud_flag = int(rng.random() < p_fraud*0.25)

        rows.append({
            "text": text,
            "emotion": emotion,
            "amount": amount,
            "channel": channel,
            "n_prev_tx_7d": n_prev,
            "churned": churned,
            "fraud_flag": fraud_flag
        })
    return pd.DataFrame(rows)

def split_train_test(df: pd.DataFrame, test_size: float=0.2, seed:int=RNG_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ntest = int(len(df)*test_size)
    return df.iloc[:-ntest].reset_index(drop=True), df.iloc[-ntest:].reset_index(drop=True)
