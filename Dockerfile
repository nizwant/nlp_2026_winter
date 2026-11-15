# Client, evaluation code and tests. The model itself is served by vLLM in a separate container
# (see docker-compose.yml) or process; this image needs no GPU.
#
#   docker build -t nlp-routing .
#   docker run --rm nlp-routing                                   # unit tests and the fake-engine pipeline
#   docker run --rm --network host -v "$PWD/outputs:/app/outputs" nlp-routing \
#       python -m routing.run configs/poc_semeval_pl.yaml         # against a vLLM server on the host
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

COPY configs ./configs
COPY prompts ./prompts
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data

CMD ["pytest", "-q"]
