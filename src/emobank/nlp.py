from __future__ import annotations
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\sáéíóúâêîôûãõç]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1,2),
        max_features=max_features
    )
