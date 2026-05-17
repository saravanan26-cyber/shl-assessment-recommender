#!/usr/bin/env bash
# ============================================================
# SHL Recommender — local setup and run
# Usage: bash run.sh
# ============================================================
set -e

echo "=== SHL Assessment Recommender ==="
echo

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
  fi
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "❌ ANTHROPIC_API_KEY not set."
  echo "   Create a .env file with: ANTHROPIC_API_KEY=sk-ant-..."
  echo "   Or: export ANTHROPIC_API_KEY=sk-ant-..."
  exit 1
fi

echo "✅ API key found"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Build catalog and index if needed
if [ ! -f "data/catalog.json" ]; then
  echo "Building catalog..."
  python3 scraper.py
fi

if [ ! -f "data/index.pkl" ]; then
  echo "Building index..."
  python3 build_index.py
fi

echo
echo "Starting server on http://localhost:8000"
echo "  Health: http://localhost:8000/health"
echo "  Chat:   POST http://localhost:8000/chat"
echo
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
