"""Minimal FastAPI proxy for a deployed A2A agent (Agent Runtime, agents-cli 1.1.0+).

The browser talks ONLY to this proxy (same origin, no CORS, no GCP creds in the
browser). The proxy authenticates with Application Default Credentials and
forwards chat to the deployed agent over the A2A protocol.
"""

import os
import uuid

import a2a.types
import google.auth
import google.auth.transport.requests
import httpx
from a2a.client import ClientConfig, ClientFactory
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google.protobuf.json_format import Parse

RESOURCE = os.environ["AGENT_ENGINE_RESOURCE_NAME"]
AGENT_DIRECTORY = os.environ.get("AGENT_DIRECTORY", "app")
LOCATION = RESOURCE.split("/locations/")[1].split("/")[0]

A2A_BASE = (
    f"https://{LOCATION}-aiplatform.googleapis.com/reasoningEngines/v1/"
    f"{RESOURCE}/api/a2a/{AGENT_DIRECTORY}"
)
A2A_CARD_URL = f"{A2A_BASE}/.well-known/agent-card.json"

_A2UI_MIME = "application/json+a2ui"

_creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)


def _auth_headers() -> dict[str, str]:
    _creds.refresh(google.auth.transport.requests.Request())
    return {
        "Authorization": f"Bearer {_creds.token}",
        "Content-Type": "application/json",
    }


app = FastAPI()


@app.exception_handler(Exception)
async def _json_errors(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={
            "parts": [{"kind": "text", "text": f"Error: {type(exc).__name__}: {exc}"}]
        },
    )


_contexts: dict[str, str] = {}
_card = None


async def _get_card(client: httpx.AsyncClient):
    global _card
    if _card is None:
        resp = await client.get(A2A_CARD_URL)
        resp.raise_for_status()
        try:
            _card = Parse(resp.text, a2a.types.AgentCard(), ignore_unknown_fields=True)
        except Exception:
            _card = a2a.types.AgentCard(**resp.json())
    return _card


def _extract_parts(parts: list) -> list[dict]:
    out: list[dict] = []
    for p in parts:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            out.append({"kind": "text", "text": text})
            continue

        raw = getattr(root, "raw", None) or getattr(root, "data", None)
        if raw is not None:
            meta = getattr(root, "metadata", None) or {}
            mime = meta.get("mimeType") if isinstance(meta, dict) else getattr(root, "mime_type", None)
            if mime == _A2UI_MIME or _A2UI_MIME in str(raw):
                out.append({"kind": "a2ui", "data": raw})
            elif isinstance(raw, (str, dict)):
                out.append({"kind": "text", "text": str(raw)})
        elif hasattr(root, "file") and getattr(root.file, "uri", None):
            out.append({"kind": "text", "text": root.file.uri})
    return out


@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    message = body.get("message", "")
    user_id = body.get("user_id") or "web-user"
    parts: list[dict] = []

    async with httpx.AsyncClient(headers=_auth_headers(), timeout=120) as client:
        card = await _get_card(client)
        factory = ClientFactory(ClientConfig(httpx_client=client))
        a2a_client = factory.create(card)

        msg = a2a.types.Message(
            message_id=str(uuid.uuid4()),
            role=a2a.types.Role.ROLE_USER,
            parts=[a2a.types.Part(text=message)],
            context_id=_contexts.get(user_id, ""),
        )
        send_req = a2a.types.SendMessageRequest(
            message=msg,
            configuration=a2a.types.SendMessageConfiguration(),
        )

        last_task = None
        got_artifact_update = False
        async for event in a2a_client.send_message(send_req):
            if isinstance(event, tuple):
                task, update = event
                if task is not None:
                    last_task = task
                    if getattr(task, "context_id", None):
                        _contexts[user_id] = task.context_id
                if isinstance(update, a2a.types.TaskArtifactUpdateEvent):
                    got_artifact_update = True
                    parts.extend(_extract_parts(update.artifact.parts))
            elif hasattr(event, "HasField"):
                if event.HasField("task"):
                    task = event.task
                    if task.context_id:
                        _contexts[user_id] = task.context_id
                elif event.HasField("artifact_update"):
                    got_artifact_update = True
                    parts.extend(_extract_parts(event.artifact_update.artifact.parts))

        if not got_artifact_update and last_task is not None:
            for artifact in getattr(last_task, "artifacts", None) or []:
                parts.extend(_extract_parts(artifact.parts))

    if not parts:
        parts = [{"kind": "text", "text": "(The agent didn't return a reply.)"}]
    return JSONResponse({"parts": parts})


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
