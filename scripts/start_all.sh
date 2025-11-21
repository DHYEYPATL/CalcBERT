#!/usr/bin/env bash
# Start all services for CalcBERT demo
# Usage: ./scripts/start_all.sh

set -e

echo "============================================================"
echo "Starting CalcBERT Services"
echo "============================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "============================================================"
    echo "Shutting down services..."
    echo "============================================================"
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${BLUE}Starting Backend (FastAPI)...${NC}"
(cd "$(dirname "$0")/.." && python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload) &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
echo "  API: http://127.0.0.1:8000"
echo "  Docs: http://127.0.0.1:8000/docs"

# Wait for backend to be ready
echo ""
echo "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Backend failed to start in 30 seconds"
        cleanup
    fi
    sleep 1
done

# Start UI (if exists)
echo ""
if [ -d "ui" ] && [ -f "ui/app.py" ]; then
    echo -e "${BLUE}Starting UI (Streamlit)...${NC}"
    (cd ui && streamlit run app.py --server.port 8501 --server.headless true) &
    UI_PID=$!
    echo -e "${GREEN}✓ UI started (PID: $UI_PID)${NC}"
    echo "  UI: http://localhost:8501"
else
    echo "UI not found (ui/app.py), skipping..."
fi

echo ""
echo "============================================================"
echo -e "${GREEN}All services running!${NC}"
echo "============================================================"
echo "Backend API: http://127.0.0.1:8000"
echo "API Docs: http://127.0.0.1:8000/docs"
if [ -d "ui" ] && [ -f "ui/app.py" ]; then
    echo "UI: http://localhost:8501"
fi
echo ""
echo "Press Ctrl+C to stop all services"
echo "============================================================"

# Wait for all background jobs
wait
