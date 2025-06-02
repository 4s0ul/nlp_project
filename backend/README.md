# Readme


```bash
docker build -f .ci/Dockerfile . -t stemming 
```

# Install

```bash 
uv sync
```

## Setup

```bash
source .venv/bin/activate.fish

python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
```

## Run

```bash
source .venv/bin/activate.fish
python main.py 
```