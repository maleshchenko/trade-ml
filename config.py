"""Central configuration, override with environment variables or edit here."""

import os

# Paths (all relative to DATA_DIR so the Docker volume keeps everything together)
DATA_DIR = os.environ.get("DATA_DIR", "data")
DATA_CSV = os.path.join(DATA_DIR, "data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pt")

# Trading pair / interval
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
INTERVAL = os.environ.get("INTERVAL", "1m")

# How many 1-min candles to download for training
TOTAL_POINTS = int(os.environ.get("TOTAL_POINTS", 10_000))
