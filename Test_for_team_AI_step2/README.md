# Step 2 — Interactive Chat About Video Highlights

**Backend:** FastAPI
**Frontend:** React (Vite)
**DB:** PostgreSQL + pgvector (re-uses the DB populated in **Step 1**)

## Run
1) Copy environment file: cp backend/.env.example backend/.env
2) docker compose build backend
3) docker compose build frontend
4) docker compose up -d backend
5) docker compose up -d frontend

## Chat Architecture
U[User] --> UI[Frontend (React/Vite)]
UI -->|POST /api/chat/ask| API[Backend (FastAPI)]
API -->|Embed query| EMB[SentenceTransformer]
API -->|Vector/Keyword search| DB[(Postgres + pgvector)]
DB --> API
API -->|Answer + Matches (JSON)| UI

# UI: http://localhost:8080
# Health: http://localhost:8000/health
# Ask:
curl -X POST http://localhost:8000/api/chat/ask \ 
-H "Content-Type: application/json" \
-d '{"query":"What occurred in the videos?","top_k":5}'

Make sure your DB already has data from Step 1 (tables created here if missing).
