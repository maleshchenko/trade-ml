#!/usr/bin/env bash
# deploy.sh — build, train, and start the signal services on a remote server
#
# Usage:
#   ./deploy.sh             # first-time full setup + start
#   ./deploy.sh --retrain   # re-download data, retrain, restart services
#   ./deploy.sh --logs      # tail API + live logs
#   ./deploy.sh --restart   # restart running services without retraining
set -euo pipefail

COMPOSE="docker compose"

case "${1:-}" in
  --logs)
    $COMPOSE logs -f api live
    exit 0
    ;;
  --restart)
    $COMPOSE restart api live
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

echo "==> Starting services..."
$COMPOSE up -d api live

echo ""
echo "✓ Services are running."
echo "  Signal API:  http://localhost:8000/signal"
echo "  API docs:    http://localhost:8000/docs"
echo "  Health:      http://localhost:8000/health"
echo ""
echo "  Logs:        ./deploy.sh --logs"
echo "  Stop:        docker compose stop api live"
echo "  Retrain:     ./deploy.sh --retrain"