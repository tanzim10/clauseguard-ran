"""Shared RCA orchestrator used by FastAPI and MCP (Week 7+). Placeholder only."""

from typing import Any


class RcaPipeline:
    """Single core pipeline — routes/MCP must call this, not duplicate logic."""

    def search_specs(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Retrieve spec chunks. Not implemented in scaffold."""
        raise NotImplementedError("Week 2/7 — search_specs")

    def grounded_query(self, question: str) -> dict[str, Any]:
        """Grounded QA with citations or abstain. Not implemented in scaffold."""
        raise NotImplementedError("Week 3 — grounded_query")

    def run_rca(self, scenario: dict[str, Any] | str) -> dict[str, Any]:
        """Full multimodal RCA. Not implemented in scaffold."""
        raise NotImplementedError("Week 7 — run_rca")
