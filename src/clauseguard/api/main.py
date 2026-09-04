"""FastAPI app factory. Only /health is fully implemented in scaffold."""

from fastapi import FastAPI

from clauseguard.api.routes import evaluate, health, query, rca, search
from clauseguard.config import get_settings
from clauseguard.logging import setup_logging


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(
        title="ClauseGuardRAN",
        description="Evidence-backed O-RAN fault diagnosis (scaffold)",
        version="0.1.0",
    )
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(query.router)
    app.include_router(rca.router)
    app.include_router(evaluate.router)
    return app


app = create_app()


def run() -> None:
    """Console entrypoint for uvicorn."""
    import uvicorn

    uvicorn.run(
        "clauseguard.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
