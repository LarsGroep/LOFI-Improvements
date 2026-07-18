# LOFI Scout + Agentic OS — one image, two commands.
# Build from the repo root:   docker build -f deploy/Dockerfile -t lofi .
# Run the OS:                 docker run --env-file .env lofi
# Run the app:                docker run --env-file .env -p 8501:8501 lofi \
#                               streamlit run scout/app.py --server.address 0.0.0.0
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: xgboost runtime (the growth model's predictions.csv loader is
# pure pandas, but retraining inside the container should also work).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# venue-private state lives on volumes (see docker-compose.yml)
VOLUME ["/app/predict/data", "/app/osk/data"]

CMD ["python", "-m", "osk", "loop"]
