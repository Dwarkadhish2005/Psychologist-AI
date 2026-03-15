"""
FastAPI application entry point — Psychologist AI Phase 6
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import users, sessions, stream, analytics, therapists


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

app = FastAPI(
    title="Psychologist AI API",
    description="Real-time multi-modal psychological state analysis",
    version="6.0.0",
)

# Allow the Vite dev server (and any local origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router,      prefix="/api/users",      tags=["users"])
app.include_router(sessions.router,   prefix="/api/sessions",   tags=["sessions"])
app.include_router(analytics.router,  prefix="/api/analytics",  tags=["analytics"])
app.include_router(therapists.router, prefix="/api/therapists", tags=["therapists"])
app.include_router(stream.router,     tags=["stream"])


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "Psychologist AI API v6.0.0"}


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
