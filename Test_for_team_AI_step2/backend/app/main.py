from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.api.chat import router as chat_router

from dotenv import load_dotenv
load_dotenv() 

app = FastAPI(title="Video Highlight Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(chat_router, prefix="/chat", tags=["chat"])

#app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
# if __name__ == "__main__":
#     import os
#     import uvicorn
#     uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
