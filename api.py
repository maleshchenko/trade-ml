"""FastAPI service — exposes the current trading signal via HTTP.

Endpoints:
    GET /signal   → current signal, price, and per-class probabilities
    GET /health   → liveness check

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

    # or via deploy.sh / docker compose:
    docker compose up -d api
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from trade_model import (
    LSTMModel,
    load_checkpoint,
    fetch_latest_klines,
    format_klines,
    add_features,
    normalize_live_features,
    FEATURE_COLS,
    SEQ_LEN,
    REALTIME_SYMBOL,
    REALTIME_INTERVAL,
)

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Global model state (loaded once at startup) ──────────────────────────────

_model: Optional[LSTMModel] = None
_means: Optional[dict] = None
_stds: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _means, _stds
    logger.info("Loading model checkpoint…")
    _model = LSTMModel()
    _means, _stds = load_checkpoint(_model)
    _model.eval()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="trade-ml signal API", lifespan=lifespan)


# ── Response schemas ─────────────────────────────────────────────────────────

class SignalResponse(BaseModel):
    signal: str                        # "long" | "short" | "neutral"
    price: float                       # latest close price
    probabilities: Dict[str, float]    # {"neutral": …, "long": …, "short": …}
    symbol: str
    interval: str
    timestamp: str                     # ISO-8601 UTC


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/signal", response_model=SignalResponse)
def get_signal(
    symbol: str = REALTIME_SYMBOL,
    interval: str = REALTIME_INTERVAL,
):
    """Fetch the latest candles from Binance and return the model's current signal."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        raw = fetch_latest_klines(symbol, interval, limit=100)
    except Exception as exc:
        logger.error("Binance fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Binance fetch failed: {exc}")

    df = format_klines(raw)
    df = add_features(df)
    df = normalize_live_features(df, _means, _stds)

    if len(df) < SEQ_LEN:
        raise HTTPException(status_code=503, detail="Not enough candle data to generate a signal")

    x = torch.tensor(
        df[FEATURE_COLS].iloc[-SEQ_LEN:].values,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        probs = _model(x)[0].numpy()

    labels = ["neutral", "long", "short"]
    signal = labels[int(np.argmax(probs))]
    price = float(df["close"].iloc[-1])

    return SignalResponse(
        signal=signal,
        price=price,
        probabilities={label: round(float(p), 4) for label, p in zip(labels, probs)},
        symbol=symbol,
        interval=interval,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)