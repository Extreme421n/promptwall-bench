"""FastAPI app exposing the tool registry over HTTP.

Phase 1E endpoints:

    GET  /health
    GET  /tools                       — list every registered tool with metadata
    GET  /tools/{tool_name}           — describe one tool (incl. JSON Schemas)
    POST /tools/{tool_name}/execute   — run a tool through the ToolExecutor

The execute endpoint accepts an optional ``trace_id``; when omitted, the
executor creates a synthetic chat session + trace so the invocation is still
captured in ``tool_invocations``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_session
from app.llm import get_provider
from app.services import ChatService, ToolExecutor
from app.tools import default_registry
from app.tools.base import ToolNotFoundError

app = FastAPI(
    title="PromptWall Benchmark API",
    description="Phase 1E: tool registry + trace-logged execution.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ToolSummary(BaseModel):
    name: str
    description: str
    domain: str
    risk_level: str
    read_only: bool


class ToolDescription(ToolSummary):
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class ListToolsResponse(BaseModel):
    count: int
    tools: list[ToolSummary]


class ExecuteToolRequest(BaseModel):
    trace_id: Optional[int] = Field(
        default=None,
        description="Existing trace id. When omitted, the executor creates a temporary trace.",
    )
    input: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolResponse(BaseModel):
    tool_name: str
    success: bool
    output: Optional[dict[str, Any]]
    error_type: Optional[str]
    error_message: Optional[str]
    latency_ms: int
    evidence_id: Optional[str]


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Existing chat session uuid; created if missing.")
    customer_id: Optional[int] = None
    mode: Literal[
        "baseline",
        "promptwall_candidate_shadow",
        "promptwall_enforced",
    ] = "baseline"
    message: str = Field(min_length=1)
    model: str = Field(default="mock")
    metadata: Optional[dict[str, Any]] = None


class ChatToolCallSummary(BaseModel):
    name: str
    arguments: dict[str, Any]
    success: bool
    evidence_id: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    latency_ms: int


class ChatResponse(BaseModel):
    answer: str
    trace_id: int
    session_id: str
    tools_called: list[ChatToolCallSummary]
    evidence_ids: list[str]
    latency_ms: int
    estimated_cost_usd: Decimal


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools", response_model=ListToolsResponse, tags=["tools"])
def list_tools() -> ListToolsResponse:
    tools = [
        ToolSummary(
            name=t.name,
            description=t.description,
            domain=t.domain,
            risk_level=t.risk_level,
            read_only=t.read_only,
        )
        for t in default_registry.list_tools()
    ]
    return ListToolsResponse(count=len(tools), tools=tools)


@app.get("/tools/{tool_name}", response_model=ToolDescription, tags=["tools"])
def describe_tool(tool_name: str) -> ToolDescription:
    try:
        tool = default_registry.get(tool_name)
    except ToolNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    return ToolDescription(
        name=tool.name,
        description=tool.description,
        domain=tool.domain,
        risk_level=tool.risk_level,
        read_only=tool.read_only,
        input_schema=tool.input_schema.model_json_schema(),
        output_schema=tool.output_schema.model_json_schema(),
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    body: ChatRequest,
    session: Session = Depends(get_session),
) -> ChatResponse:
    try:
        provider = get_provider(body.model)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    service = ChatService(session=session, provider=provider)
    result = service.chat(
        message=body.message,
        session_uuid=body.session_id,
        customer_id=body.customer_id,
        mode=body.mode,
        metadata=body.metadata,
        model=body.model,
    )
    return ChatResponse(
        answer=result.answer,
        trace_id=result.trace_id,
        session_id=result.session_uuid,
        tools_called=[
            ChatToolCallSummary(
                name=t.name,
                arguments=t.arguments,
                success=t.success,
                evidence_id=t.evidence_id,
                error_type=t.error_type,
                error_message=t.error_message,
                latency_ms=t.latency_ms,
            )
            for t in result.tools_called
        ],
        evidence_ids=result.evidence_ids,
        latency_ms=result.latency_ms,
        estimated_cost_usd=result.estimated_cost_usd,
    )


@app.post(
    "/tools/{tool_name}/execute",
    response_model=ExecuteToolResponse,
    tags=["tools"],
)
def execute_tool(
    tool_name: str,
    body: ExecuteToolRequest,
    session: Session = Depends(get_session),
) -> ExecuteToolResponse:
    # 404 for unknown tools so clients can distinguish from input errors.
    if tool_name not in default_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no tool named {tool_name!r}",
        )

    executor = ToolExecutor(session)
    result = executor.execute_tool(
        trace_id=body.trace_id,
        tool_name=tool_name,
        input_json=body.input,
    )
    return ExecuteToolResponse(
        tool_name=result.tool_name,
        success=result.success,
        output=result.output,
        error_type=result.error_type,
        error_message=result.error_message,
        latency_ms=result.latency_ms,
        evidence_id=result.evidence_id,
    )
