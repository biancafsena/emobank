.PHONY: data train train_emotion train_risk api test

data:
	python scripts/make_data.py

train_emotion:
	python scripts/train_emotion.py

train_risk:
	python scripts/train_risk.py

train: data train_emotion train_risk

api:
	python scripts/serve.py

test:
	pytest -q
