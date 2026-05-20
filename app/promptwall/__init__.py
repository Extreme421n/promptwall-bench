"""PromptWall: shadow analysis (Phase 3A) and, eventually, enforcement.

The Phase 3A surface is intentionally minimal: a single ``PromptWallCandidateAnalyzer``
class that emits a structured :class:`CandidateDecision` for any user
message. Nothing in this module affects the chatbot's behaviour — it only
predicts and logs.
"""

from app.promptwall.analyzer import (
    CandidateDecision,
    PromptWallCandidateAnalyzer,
)
from app.promptwall.router import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    EnforcementDecision,
    PromptWallRouter,
)

__all__ = [
    "CandidateDecision",
    "PromptWallCandidateAnalyzer",
    "EnforcementDecision",
    "PromptWallRouter",
    "DEFAULT_CONFIDENCE_THRESHOLD",
]
