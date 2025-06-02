# Use official Python 3.12 slim image to reduce size
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=false

# Set working directory
WORKDIR /app

# Install system dependencies (for psycopg2, hdbscan, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Copy dependency files
COPY ./pyproject.toml /app/pyproject.toml

# Install dependencies and spaCy models
RUN poetry install --no-root && \
    poetry run python -m spacy download en_core_web_md && \
    poetry run python -m spacy download en_core_web_sm && \
    poetry run python -m spacy download ru_core_news_sm && \
    poetry run python -m spacy download ru_core_news_md && \
    poetry run python -c "import nltk; nltk.download('stopwords')"

# Copy project code (only necessary files)
COPY ./stemming /app/stemming

# Expose port
EXPOSE 8000

# Run the app using uvicorn
CMD [ "poetry", "run", "uvicorn", "stemming.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
