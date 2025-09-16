# EmoBank — Análise de Emoções em Transações  

![Python](https://img.shields.io/badge/Python-3.10+-blue) 
![FastAPI](https://img.shields.io/badge/FastAPI-API-green) 
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SHAP-orange) 
![R](https://img.shields.io/badge/R-tidyverse-blueviolet) 
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

**NLP + Speech-to-Text + Detecção de Emoções** para prever **risco de churn** ou **fraude emocional** em bancos.  
Projeto profissional, pronto para **GitHub/LinkedIn** e para rodar no **VS Code** ou **Docker**.  

---

## 💡 Por que esse projeto chama atenção?
- Junta **comportamento humano** (emoções em voz/texto) com **eventos financeiros** (transação, canal, valor).  
- Mostra **stack moderna**: FastAPI, scikit-learn, SHAP, testes, Docker e um **R Markdown** visual.  
- Foco em **explicabilidade** (com SHAP para o modelo de risco) e métricas claras.  

---

## 📂 Arquitetura de pastas

emobank/
├─ src/emobank/
│ ├─ data.py # Geração de dados sintéticos & schemas
│ ├─ nlp.py # Limpeza, tokenização, vetorização TF-IDF
│ ├─ emotion_model.py # Treino/classificação de emoções (texto)
│ ├─ risk_model.py # Modelo de risco de churn/fraude (usa emoção + metadados)
│ ├─ api.py # FastAPI com /predict_text, /predict_audio*, /explain
│ └─ utils.py # utilidades (carregamento de artefatos etc.)
├─ scripts/
│ ├─ make_data.py # gera dataset sintético
│ ├─ train_emotion.py # treina classificador de emoção
│ ├─ train_risk.py # treina modelo de risco com SHAP
│ └─ serve.py # sobe a API (uvicorn)
├─ artifacts/ # modelos salvos (joblib/json)
├─ data/ # CSVs sintéticos
├─ tests/ # pytest
├─ r/
│ ├─ analysis.Rmd # EDA e métricas em R
│ └─ install_packages.R
├─ docker/Dockerfile
├─ .vscode/launch.json
├─ .vscode/settings.json
├─ requirements.txt
├─ Makefile
├─ .gitignore
└─ README.md


> *Speech-to-Text*: endpoint de áudio tenta usar **faster-whisper** se instalado.  
> Se não estiver, retorna mensagem amigável (o projeto continua funcional via texto).  

---

## ⚙️ Como rodar (local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install  # opcional

```

### 1) Gerar dados e treinar
```bash
make data
make train  # treinando emoção + risco
```

### 2) Subir API
```bash
make api
# Docs: http://localhost:8000/docs
```

### 3) Exemplos de uso
**Predição via texto**
```bash
curl -X POST http://localhost:8000/predict_text -H "Content-Type: application/json" -d '{
  "text": "Já liguei 3 vezes e ninguém resolve meu problema! Estou muito irritado.",
  "amount": 1200.50, "channel": "pix", "n_prev_tx_7d": 18
}'
```
**Explicação do risco (SHAP)**
```bash
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d '{
  "text": "A experiência foi ótima, parabéns pela rapidez!",
  "amount": 89.9, "channel": "card", "n_prev_tx_7d": 2
}'
```

> **Áudio**: `POST /predict_audio` aceita `.wav` mono 16kHz. Se `faster-whisper` não estiver instalado, o endpoint te orienta como habilitar.


## 🐳 Docker

```bash
docker build -f docker/Dockerfile -t emobank:latest .
docker run --rm -p 8000:8000 emobank:latest
```

## 📊 R (opcional, para EDA e métricas)

Instalar pacotes e knit:
```bash
Rscript r/install_packages.R
# Em R:
# rmarkdown::render("r/analysis.Rmd")
```


## Stack técnica
- **Python** 3.10+: FastAPI, scikit-learn, pandas, numpy, shap, joblib
- **Whisper opcional**: faster-whisper (GPU/CPU), soundfile
- **R**: tidyverse, yardstick, rmarkdown
- Testes com **pytest**

## Licença
MIT

---

👉 Esse é o `README.md` completo já polido para impressionar recrutadores.  
Quer que eu também prepare o **.gitignore** personalizado (Python + R + macOS) pra você já colar no projeto antes do commit?
