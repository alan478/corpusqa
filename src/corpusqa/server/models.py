"""Wire models for the HTTP interface in CONTRACT.md."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# Strict: a number or null is a bad request, not a question to coerce.
QuestionText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class AskRequest(BaseModel):
    """POST /ask body. Unknown extra fields are ignored."""

    question: QuestionText


class AskResponse(BaseModel):
    """POST /ask body: exactly CONTRACT §3's two fields (user decision 2026-10-17).

    The trace id is not in the body: the server sends it in the `TRACE_ID_HEADER` header.
    """

    answer: str = Field(min_length=1)
    document_ids: list[str]


class HealthResponse(BaseModel):
    """GET /health body. The grader reads only the status code."""

    status: Literal["starting", "ok", "failed"]
