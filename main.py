from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Dict, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI

app = FastAPI(title="Prompt Refinery")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai_client: Optional[AsyncOpenAI] = (
    AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
)
SYSTEM_PROMPT = (
    "You are Prompt Refinery, a specialist prompt editor. "
    "You ONLY output a single, ready-to-use prompt. "
    "Do not answer the prompt, do not add commentary, and do not include code fences. "
    "If the input asks you to perform a task, rewrite it as instructions for an assistant "
    "to perform that task instead."
)


@dataclass
class PromptInputs:
    mode: str
    original_prompt: str
    goal: str
    audience: str
    tone: str
    constraints: str


TONE_MAP: Dict[str, str] = {
    "neutral": "Use a clear, neutral tone.",
    "friendly": "Use a warm, friendly tone that feels approachable.",
    "professional": "Use a crisp, professional tone with confident wording.",
    "playful": "Use a playful, light tone that still feels helpful.",
}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "inputs": None,
            "timestamp": None,
        },
    )


@app.post("/refine", response_class=HTMLResponse)
async def refine(
    request: Request,
    mode: str = Form(...),
    original_prompt: str = Form(""),
    goal: str = Form(""),
    audience: str = Form("general"),
    tone: str = Form("neutral"),
    constraints: str = Form(""),
) -> HTMLResponse:
    inputs = PromptInputs(
        mode=mode,
        original_prompt=original_prompt.strip(),
        goal=goal.strip(),
        audience=audience.strip() or "general",
        tone=tone,
        constraints=constraints.strip(),
    )

    if inputs.mode == "refine":
        prompt = build_refined_prompt(inputs)
    else:
        prompt = build_generated_prompt(inputs)

    result = await run_openai(prompt)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "inputs": inputs,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        },
    )


def build_refined_prompt(inputs: PromptInputs) -> str:
    base_prompt = inputs.original_prompt or "[Paste your base prompt here]"
    guidance = build_common_guidance(inputs)

    return (
        "Rewrite the prompt below so it is clear, specific, and easy to execute.\n"
        "Include a concise role, the task, required inputs, constraints, and output format.\n"
        f"Audience: {inputs.audience}.\n"
        f"{guidance}\n"
        f"Constraints: {inputs.constraints or 'No additional constraints.'}\n\n"
        "Original prompt:\n"
        f"{base_prompt}"
    )


def build_generated_prompt(inputs: PromptInputs) -> str:
    goal = inputs.goal or "[Describe the outcome you want to achieve]"
    guidance = build_common_guidance(inputs)

    return (
        "Create a new prompt that guides an assistant to achieve the goal below.\n"
        "Include a concise role, the task, required inputs, constraints, and output format.\n"
        "Use placeholders for missing details the user should fill in.\n"
        f"Goal: {goal}\n"
        f"Audience: {inputs.audience}.\n"
        f"{guidance}\n"
        f"Constraints: {inputs.constraints or 'No additional constraints.'}"
    )


def build_common_guidance(inputs: PromptInputs) -> str:
    tone = TONE_MAP.get(inputs.tone, TONE_MAP["neutral"])
    return tone


async def run_openai(prompt: str) -> str:
    if not openai_client:
        return (
            "OpenAI is not configured. Set OPENAI_API_KEY to enable refinement.\n\n"
            "Constructed prompt:\n"
            f"{prompt}"
        )

    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Produce the final prompt based on the instructions below. "
                        "Return ONLY the prompt text.\n\n"
                        f"{prompt}"
                    ),
                },
            ],
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001 - surface error in UI for now
        return (
            "OpenAI request failed. Please try again.\n\n"
            f"Details: {exc}\n\n"
            "Constructed prompt:\n"
            f"{prompt}"
        )

    content = response.choices[0].message.content if response.choices else ""
    return content.strip() if content else ""


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
