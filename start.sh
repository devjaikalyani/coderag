#!/bin/bash
# CodeRAG — start backend + frontend together
# Usage: ./start.sh

# Intel Mac OpenMP fix
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ No .env file. Run: cp .env.example .env and add GROQ_API_KEY"
    exit 1
fi

# Kill anything already on these ports
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true
sleep 1

# Cleanup on Ctrl+C
cleanup() {
    echo ""
    echo "Stopping CodeRAG..."
    kill $API_PID $UI_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# 1. Start API
echo "🚀 Starting API on http://localhost:8001 ..."
uvicorn src.api.main:app \
    --port 8001 \
    --workers 1 \
    --reload \
    --reload-dir src \
    --log-level warning &
API_PID=$!

# 2. Wait until API is healthy (max 30s)
echo "   Waiting for API to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "   ✅ API ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ API failed to start. Check for errors above."
        kill $API_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# 3. Start Streamlit AFTER API is ready
echo "🎨 Starting UI on http://localhost:8501 ..."
streamlit run ui/app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.runOnSave true \
    --server.fileWatcherType auto \
    --browser.gatherUsageStats false \
    --browser.serverAddress localhost &
UI_PID=$!

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  ✅ CodeRAG running!                  ║"
echo "║  UI  → http://localhost:8501          ║"
echo "║  API → http://localhost:8001          ║"
echo "║  Press Ctrl+C to stop both            ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  Changes to src/ → API auto-reloads"
echo "  Changes to ui/  → UI auto-reloads"
echo ""

wait $API_PID $UI_PID