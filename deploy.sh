#!/usr/bin/env bash
# deploy.sh — build, train, and start the live signal service on a remote server
#
# Usage:
#   ./deploy.sh             # first-time full setup + start
#   ./deploy.sh --retrain   # re-download data, retrain, restart live service
#   ./deploy.sh --logs      # tail live service logs
#   ./deploy.sh --restart   # restart live service without retraining
set -euo pipefail

COMPOSE="docker compose"

case "${1:-}" in
  --logs)
    $COMPOSE logs -f live
    exit 0
    ;;
  --restart)
    $COMPOSE restart live
    exit 0
    ;;
esac

echo "==> Creating data directory..."
mkdir -p data

echo "==> Building Docker image..."
$COMPOSE build

echo "==> Downloading historical data..."
$COMPOSE --profile setup run --rm dataloader

echo "==> Training model..."
if [ "${1:-}" = "--retrain" ]; then
  $COMPOSE --profile setup run --rm train python train.py --retrain
else
  $COMPOSE --profile setup run --rm train
fi

echo "==> Evaluating model..."
$COMPOSE --profile setup run --rm evaluate

echo "==> Starting live signal service..."
$COMPOSE up -d live

echo ""
echo "✓ Live signal service is running."
echo "  Logs:    docker compose logs -f live"
echo "  Stop:    docker compose stop live"
echo "  Retrain: ./deploy.sh --retrain"
