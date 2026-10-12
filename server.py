"""Typed decisions over HTTP, served by a Laya decision model on one GPU.

Runs inside the container image that deploy.py builds for a Chalk ScalingGroup.

    POST /decide   {"state": <text or JSON record>, "questions": {"<id>": {"type": ..., ...}}}
    GET  /status   readiness and model info
    GET  /         a small standalone playground (index.html)

A question is one of three types, all answered in a single forward pass:
    choice  {"type": "choice", "instructions": "...", "criteria": {"<option>": "<description>", ...}}
    noul    {"type": "noul",   "instructions": "..."}            -> P(true)
    score   {"type": "score",  "instructions": "...", "criteria": ["<level 0>", "<level 1>", ...]}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import laya
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

MODEL_ID = os.environ.get("DECISIONS_MODEL", "michaelfeil/laya-typed-decisions")
MODEL_REVISION = os.environ.get("DECISIONS_MODEL_REVISION") or None
PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(__file__).with_name("index.html")
# Browsers on these origins may call /decide directly. Anything else can still use the API
# from a server; CORS only governs what a web page is allowed to read.
ALLOWED_ORIGIN_REGEX = os.environ.get(
    "DECISIONS_ALLOWED_ORIGIN_REGEX",
    r"https://([a-z0-9-]+\.)*chalk\.(ai|dev)|http://localhost(:\d+)?",
)

# Public-demo limits: one request is one forward pass, so these bound its cost.
MAX_STATE_CHARS = 6000
MAX_QUESTIONS = 8
MAX_OPTIONS = 20
MAX_TEXT_CHARS = 400
RATE_LIMIT_PER_MINUTE = int(os.environ.get("DECISIONS_RATE_LIMIT_PER_MINUTE", "60"))

log = logging.getLogger("decisions")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def check_criteria(self) -> Question:
        if self.type == "choice" and not isinstance(self.criteria, dict):
            raise ValueError("a choice question needs criteria as {option: description}")
        if self.type == "score" and not isinstance(self.criteria, list):
            raise ValueError("a score question needs criteria as an ordered list of levels")
        if self.type == "noul" and self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError("noul criteria, when given, are {false: ..., true: ...}")
        items = list(self.criteria.items()) if isinstance(self.criteria, dict) else self.criteria or []
        if self.type != "noul" and not 2 <= len(items) <= MAX_OPTIONS:
            raise ValueError(f"{self.type} questions take 2 to {MAX_OPTIONS} options")
        for item in items:
            for text in item if isinstance(item, tuple) else (item,):
                if len(text) > MAX_TEXT_CHARS:
                    raise ValueError(f"criteria text is limited to {MAX_TEXT_CHARS} characters")
        return self


class DecideRequest(BaseModel):
    # Free text, or a record (ticket, invoice, alert, trace) that the model reads as JSON.
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @model_validator(mode="after")
    def check_request(self) -> DecideRequest:
        size = len(self.state) if isinstance(self.state, str) else len(json.dumps(self.state))
        if not 1 <= size <= MAX_STATE_CHARS:
            raise ValueError(f"state must be 1 to {MAX_STATE_CHARS} characters (as JSON for a record)")
        for qid in self.questions:
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", qid):
                raise ValueError(f"question id {qid!r} must be 1-64 letters, digits, _ or -")
        return self


class RateLimiter:
    """Sliding one-minute window per client address."""

    def __init__(self, per_minute: int) -> None:
        super().__init__()
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self.hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


def client_key(request: Request) -> str:
    # Behind the scaling group's gateway, the client address is the first X-Forwarded-For hop.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


class Decider:
    def __init__(self) -> None:
        super().__init__()
        t0 = time.monotonic()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = laya.load(MODEL_ID, device=self.device, revision=MODEL_REVISION)
        # The agent is not documented as thread-safe, and one device runs one pass at a time anyway.
        self.lock = threading.Lock()
        warm = {"q": {"type": "noul", "instructions": "Is this a warmup?"}}
        for _ in range(3):
            self.agent.predict("warmup", warm)
        log.info("loaded %s on %s in %.1fs", MODEL_ID, self.device, time.monotonic() - t0)

    def decide(self, req: DecideRequest) -> dict[str, Any]:
        questions = {qid: q.model_dump(exclude_none=True) for qid, q in req.questions.items()}
        with self.lock:
            t0 = time.perf_counter()
            result = self.agent.predict(req.state, questions)
            if self.device == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000
        return {
            "answers": result["answers"],
            "usage": result.get("usage", {}),
            "infer_ms": round(infer_ms, 1),
        }


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # uvicorn accepts no connections until this returns, so the replica goes ready only once
    # the model is resident and warm.
    state["decider"] = await asyncio.to_thread(Decider)
    state["limiter"] = RateLimiter(RATE_LIMIT_PER_MINUTE)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
    max_age=600,
)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/status")
async def status() -> JSONResponse:
    decider: Decider = state["decider"]
    return JSONResponse({"ok": True, "model": MODEL_ID, "device": decider.device})


@app.post("/decide")
async def decide(req: DecideRequest, request: Request) -> JSONResponse:
    limiter: RateLimiter = state["limiter"]
    if not limiter.allow(client_key(request)):
        raise HTTPException(status_code=429, detail="Too many requests; try again in a minute.")
    decider: Decider = state["decider"]
    return JSONResponse(await asyncio.to_thread(decider.decide, req))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
