FROM python:3.10-slim as builder

RUN pip install poetry==1.6.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY ./pyproject.toml ./poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

FROM python:3.10-slim as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app

COPY ./src/rest ./rest/
COPY ./src/ml_models ./ml_models/
COPY ./src/utils/ ./utils/
COPY ./src/main.py ./main.py
COPY ./src/.env ./.env
COPY ./src/llm_ds_faulty.csv ./llm_ds_faulty.csv

ENTRYPOINT ["python3", "main.py"]