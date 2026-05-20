"""BenchmarkRunner: replay eval cases through the chat service and score them.

The runner uses the internal :class:`ChatService` directly (no HTTP overhead),
which means a benchmark of N cases triggers exactly N chat sessions and writes
the same trace artifacts as a production /chat call. Concurrency is opt-in via
``workers``; each worker holds its own SQLAlchemy session.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.eval.scorer import (
    CaseScore,
    ToolInvocationSummary,
    aggregate_metrics,
    score_case,
)
from app.llm import get_provider
from app.models import EvaluationResult, EvaluationRun, ToolInvocation
from app.services import ChatService


@dataclass
class RunSummary:
    run_id: int
    name: str
    total_cases: int
    metrics: dict[str, Any]
    scores: list[CaseScore] = field(default_factory=list)


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file of eval cases into a list of dicts."""
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


@dataclass
class _CaseOutcome:
    case: dict[str, Any]
    trace_id: Optional[int]
    answer: str
    actual_invocations: list[ToolInvocationSummary]
    latency_ms: int
    error: Optional[str]


class BenchmarkRunner:
    """Run a benchmark over a list of eval cases and persist the results."""

    def __init__(
        self,
        engine: Engine,
        *,
        mode: str = "baseline",
        model: str = "mock",
        workers: int = 1,
    ) -> None:
        self.engine = engine
        self.mode = mode
        self.model = model
        self.workers = max(1, workers)
        self._Session = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False, future=True
        )

    # ------------------------------------------------------------------

    def run(
        self,
        cases: list[dict[str, Any]],
        *,
        eval_file: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> RunSummary:
        run = self._create_run(cases, eval_file=eval_file, run_name=run_name)
        run_id = run.id

        outcomes: list[_CaseOutcome] = []
        if self.workers == 1:
            for case in cases:
                outcomes.append(self._run_one(case))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._run_one, c): c for c in cases}
                outcomes = [f.result() for f in as_completed(futures)]
            # Preserve the input order in the persisted results.
            ordered: dict[str, _CaseOutcome] = {o.case["id"]: o for o in outcomes}
            outcomes = [ordered[c["id"]] for c in cases]

        scores: list[CaseScore] = []
        with self._Session() as session:
            for out in outcomes:
                score = score_case(
                    case_id=out.case["id"],
                    category=out.case["category"],
                    must_use_tool=out.case["must_use_tool"],
                    expected_tools=out.case.get("expected_tools", []),
                    actual_invocations=out.actual_invocations,
                    answer=out.answer,
                )
                scores.append(score)
                session.add(
                    EvaluationResult(
                        run_id=run_id,
                        case_id=out.case["id"],
                        category=out.case["category"],
                        expected_domain=out.case.get("expected_domain"),
                        risk=out.case.get("risk"),
                        trace_id=out.trace_id,
                        message=out.case["message"],
                        must_use_tool=out.case["must_use_tool"],
                        expected_tools_json=list(out.case.get("expected_tools", [])),
                        actual_tools_json=[
                            {
                                "tool_name": inv.tool_name,
                                "success": inv.success,
                                "evidence_id": inv.evidence_id,
                                "error_type": inv.error_type,
                                "error_message": inv.error_message,
                            }
                            for inv in out.actual_invocations
                        ],
                        tool_called_when_required=score.tool_called_when_required,
                        tool_skip=score.tool_skip,
                        expected_tool_hit=score.expected_tool_hit,
                        wrong_tool=score.wrong_tool,
                        missing_evidence=score.missing_evidence,
                        clarification_ok=score.clarification_ok,
                        suspicious_unsupported_claim=score.suspicious_unsupported_claim,
                        answer=out.answer,
                        latency_ms=out.latency_ms,
                        error=out.error,
                    )
                )
            metrics = aggregate_metrics(
                scores, [o.latency_ms for o in outcomes if o.latency_ms is not None]
            )
            run = session.get(EvaluationRun, run_id)
            run.metrics_json = metrics
            run.total_cases = len(cases)
            run.ended_at = datetime.now(timezone.utc)
            session.commit()

        return RunSummary(
            run_id=run_id,
            name=run.name if hasattr(run, "name") else "",
            total_cases=len(cases),
            metrics=metrics,
            scores=scores,
        )

    # ------------------------------------------------------------------

    def _create_run(
        self,
        cases: list[dict[str, Any]],
        *,
        eval_file: Optional[str],
        run_name: Optional[str],
    ) -> EvaluationRun:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        name = run_name or f"{self.mode}_{self.model}_{ts}"
        with self._Session() as session:
            run = EvaluationRun(
                name=name,
                mode=self.mode,
                model=self.model,
                eval_file=eval_file,
                total_cases=len(cases),
            )
            session.add(run)
            session.commit()
            return run

    def _run_one(self, case: dict[str, Any]) -> _CaseOutcome:
        t0 = time.perf_counter()
        try:
            provider = get_provider(self.model)
            with self._Session() as session:
                service = ChatService(session=session, provider=provider)
                result = service.chat(
                    message=case["message"],
                    customer_id=case.get("customer_id"),
                    mode=self.mode,
                    metadata={
                        "eval_case_id": case["id"],
                        "category": case.get("category"),
                    },
                    model=self.model,
                )
                actual = self._collect_actual_invocations(session, result.trace_id)
                return _CaseOutcome(
                    case=case,
                    trace_id=result.trace_id,
                    answer=result.answer,
                    actual_invocations=actual,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    error=None,
                )
        except Exception as e:  # noqa: BLE001 — capture into the row, never abort the run
            return _CaseOutcome(
                case=case,
                trace_id=None,
                answer="",
                actual_invocations=[],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"{type(e).__name__}: {e}",
            )

    @staticmethod
    def _collect_actual_invocations(
        session: Session, trace_id: int
    ) -> list[ToolInvocationSummary]:
        rows = (
            session.execute(
                select(ToolInvocation)
                .where(ToolInvocation.trace_id == trace_id)
                .order_by(ToolInvocation.id)
            )
            .scalars()
            .all()
        )
        return [
            ToolInvocationSummary(
                tool_name=r.tool_name,
                success=bool(r.success),
                evidence_id=r.evidence_id,
                error_message=r.error_message,
                error_type=None,  # we don't store it; not needed by the scorer
            )
            for r in rows
        ]
