FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py trade_model.py dataloader.py train.py evaluate.py live.py ./

# Data and model artifacts live in a mounted volume
VOLUME ["/app/data"]

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=data

CMD ["python", "live.py", "--no-sound"]
