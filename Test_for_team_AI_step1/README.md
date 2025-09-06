# Step 1: Video Processor with LLM‑Based Highlight Extraction

Python + Gemini + PostgreSQL/pgvector + Docker.

## Run:
1) Copy environment file: cp .env.example .env and set `GOOGLE_API_KEY` (AI Studio).
2) Add videos (30–90s) in `input_videos/`.
3) docker compose build app        # build the Python app image
4) docker compose up -d db         # start Postgres + pgvector
5) docker compose run --rm app python -m app.demo --input input_videos  # run app
