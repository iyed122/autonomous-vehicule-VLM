"""
vlm_reasoner.py
===============
A Vision-Language reasoning layer on top of the geometric perception/planning stack.

The classical pipeline (YOLO + BEV planner + collision-risk fusion) is geometric and
closed-vocabulary: it measures *where* obstacles are and *whether* the planned path hits
one. It cannot say "there is a construction zone ahead" or "the traffic light is red" or
explain *why* it braked. A VLM closes that gap. We feed a local VLM the **full surround
view (the 6 nuScenes cameras) together with the LiDAR bird's-eye-view render** — plus a
compact text summary of what the geometric stack already found — and ask it to:

  * caption the scene,
  * flag OPEN-VOCABULARY hazards the detector cannot (construction, emergency vehicles,
    red lights, debris, vulnerable road users, weather/road state),
  * corroborate or challenge the system's CLEAR/CAUTION/BRAKE decision, and explain it,
  * recommend an action.

Why surround + BEV instead of just the front camera? The driver already sees the front
camera; the value is in *synthesising all six views and the LiDAR top-down at once* —
something no single human view provides. VLMs can't ingest raw point clouds, so the LiDAR
is given as the rendered BEV image (which they read fine).

Backends (pluggable):
  * OllamaVLM  — talks to a local Ollama server (default). Use a VISION model such as
                 `qwen2.5vl:7b` (Qwen2.5-VL, multi-image) or `llama3.2-vision`.
                 NOTE: `qwen3:8b` is TEXT-ONLY — it will reason over the text perception
                 summary but will NOT see the images. For true camera+LiDAR reasoning use
                 a *-vl / vision model.
  * MockVLM    — returns a canned, schema-valid analysis so the whole pipeline (prompt
                 assembly, image encoding, parsing, rendering) can be tested with no GPU
                 and no server.

This module has no heavy dependencies (stdlib + optional `requests`); it does not import
torch.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
# Recommended local vision model (Qwen family, multi-image). Override as needed.
DEFAULT_MODEL = "qwen2.5vl:7b"

# Vision markers take priority (so qwen3-vl / qwen2.5vl are seen as VISION even though
# they contain "qwen3"). Otherwise these name fragments indicate a TEXT-ONLY model.
VISION_HINTS = ("vl", "vision", "llava", "minicpm-v", "moondream", "bakllava")
TEXT_ONLY_HINTS = ("qwen3:", "qwen3-coder", "llama3.1", "mistral", "phi", "gemma2",
                   "deepseek")


EMBED_HINTS = ("embed", "nomic", "bge", "minilm", "e5-", "gte-")


def is_embedding_model(model: str) -> bool:
    """Embedding models can't chat or accept images — exclude them from the demo."""
    return any(h in model.lower() for h in EMBED_HINTS)


def is_vision_model(model: str) -> bool:
    """Best-effort: can this Ollama model accept images?"""
    m = model.lower()
    if is_embedding_model(m):
        return False
    if any(h in m for h in VISION_HINTS):
        return True
    if any(h in m for h in TEXT_ONLY_HINTS):
        return False
    return True  # default: assume vision-capable


def is_cloud_model(model: str) -> bool:
    """Ollama cloud-hosted models (…-cloud / :cloud) need `ollama signin` and do not
    serve through the plain local /api/generate call reliably."""
    m = model.lower()
    return m.endswith("-cloud") or ":cloud" in m or m.endswith("cloud")


def list_ollama_models(host: str = OLLAMA_DEFAULT_HOST):
    """Return the list of locally available Ollama model names (or [] if unreachable)."""
    try:
        import requests
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Result schema
# --------------------------------------------------------------------------- #
@dataclass
class VLMAnalysis:
    caption: str = ""
    hazards: List[str] = field(default_factory=list)
    agrees_with_decision: Optional[bool] = None
    rationale: str = ""
    recommended_action: str = ""
    driver_instruction: str = ""
    model: str = ""
    used_images: bool = False
    raw: str = ""

    def to_dict(self) -> Dict:
        d = dataclasses.asdict(self)
        return d


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION = (
    "You are the high-level reasoning module of an ADAS (advanced driver-assistance "
    "system). You receive the vehicle's surround-view camera images and a top-down "
    "LiDAR bird's-eye-view (BEV) render, plus a summary of what the geometric perception "
    "and planning stack already computed. Reason over ALL views jointly."
)

OUTPUT_SCHEMA = (
    "Respond with ONLY a JSON object of exactly this shape:\n"
    "{\n"
    '  "caption": "<one-sentence description of the driving scene>",\n'
    '  "hazards": ["<short open-vocabulary hazard>", ...],\n'
    '  "agrees_with_decision": true | false,\n'
    '  "rationale": "<one or two sentences explaining the risk decision>",\n'
    '  "recommended_action": "<maintain | slow | brake | stop | proceed-with-caution>",\n'
    '  "driver_instruction": "<ONE short spoken instruction to the driver, imperative, '
    '<=12 words, e.g. \'Caution, pedestrian ahead on the right, slow down\'>"\n'
    "}\n"
    "Flag hazards a bounding-box detector would miss: construction zones, emergency "
    "vehicles, traffic-light state, road debris, animals, vulnerable road users, "
    "weather/road-surface conditions. Keep every string concise."
)


def build_prompt(context: Dict, image_labels: Optional[List[str]] = None) -> str:
    """Assemble the text prompt from the geometric stack's structured context.

    context keys (all optional):
        risk_level, nearest_obstacle_m, ttc_s, ego_speed_kmh,
        detections (dict camera->list[str] or list[str]), notes (str)
    image_labels: human-readable label per image, in the order images are sent.
    """
    lines = [SYSTEM_INSTRUCTION, ""]
    if image_labels:
        lines.append("Images provided, in order: " + ", ".join(image_labels) + ".")
        lines.append("")
    lines.append("Geometric perception/planning summary:")
    risk = context.get("risk_level", "unknown")
    lines.append(f"- collision-risk decision: {risk}")
    if context.get("nearest_obstacle_m") is not None:
        lines.append(f"- nearest in-path obstacle: {context['nearest_obstacle_m']}")
    if context.get("ttc_s") is not None:
        lines.append(f"- time-to-obstacle (TTC): {context['ttc_s']}")
    if context.get("ego_speed_kmh") is not None:
        lines.append(f"- estimated ego speed: {context['ego_speed_kmh']} km/h")
    det = context.get("detections")
    if isinstance(det, dict):
        for cam, objs in det.items():
            if objs:
                lines.append(f"- {cam} detections: {', '.join(objs)}")
    elif isinstance(det, list) and det:
        lines.append(f"- detections: {', '.join(det)}")
    if context.get("notes"):
        lines.append(f"- notes: {context['notes']}")
    lines.append("")
    lines.append(OUTPUT_SCHEMA)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class VLMBackend:
    """Interface: return the raw model text for a prompt + optional images (bytes)."""
    model = "base"
    vision = True

    def generate(self, prompt: str, images: Optional[List[bytes]] = None) -> str:
        raise NotImplementedError


class OllamaVLM(VLMBackend):
    """Local Ollama backend (https://ollama.com). Requires `ollama serve` running and
    the model pulled: e.g. `ollama pull qwen2.5vl:7b`."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_DEFAULT_HOST,
                 timeout: int = 240):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.vision = is_vision_model(model)

    def generate(self, prompt: str, images: Optional[List[bytes]] = None) -> str:
        import requests  # lazy import so the module loads without the dep
        # Use /api/chat (the current endpoint; /api/generate is deprecated and returns
        # 410 for cloud-routed models). Images attach to the user message.
        message = {"role": "user", "content": prompt}
        if self.vision and images:
            message["images"] = [base64.b64encode(b).decode("ascii") for b in images]
        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "format": "json",          # force well-formed JSON output
            "think": False,            # reasoning models (qwen3): answer directly
            "options": {"temperature": 0.2},
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "") or data.get("response", "")


class MockVLM(VLMBackend):
    """Offline backend returning a schema-valid canned analysis (for testing/demo)."""

    model = "mock"

    def __init__(self, canned: Optional[Dict] = None, vision: bool = True):
        self.vision = vision
        self._canned = canned or {
            "caption": "Urban two-way street, parked vehicles on the right, overcast.",
            "hazards": ["parked vehicles narrowing lane", "pedestrian near right curb",
                        "wet road surface"],
            "agrees_with_decision": True,
            "rationale": "A vehicle sits within the planned path at close range; "
                         "braking is appropriate.",
            "recommended_action": "brake",
            "driver_instruction": "Brake now — vehicle stopped close ahead.",
        }

    def generate(self, prompt: str, images: Optional[List[bytes]] = None) -> str:
        return json.dumps(self._canned)


def make_backend(kind: str = "ollama", model: str = DEFAULT_MODEL, **kw) -> VLMBackend:
    if kind == "mock":
        return MockVLM(**kw)
    return OllamaVLM(model=model, **kw)


# --------------------------------------------------------------------------- #
# Parsing & top-level entry point
# --------------------------------------------------------------------------- #
def _parse(raw: str) -> Dict:
    """Robustly pull the JSON object out of a model response."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(raw[s:e + 1])
        except Exception:
            pass
    return {}


def analyze(backend: VLMBackend, context: Dict,
            images: Optional[List[bytes]] = None,
            image_labels: Optional[List[str]] = None) -> VLMAnalysis:
    """Run one VLM reasoning pass and return a parsed VLMAnalysis."""
    prompt = build_prompt(context, image_labels)
    raw = backend.generate(prompt, images=images)
    d = _parse(raw)
    hazards = d.get("hazards", [])
    if isinstance(hazards, str):
        hazards = [hazards]
    return VLMAnalysis(
        caption=str(d.get("caption", "")).strip(),
        hazards=[str(h) for h in hazards],
        agrees_with_decision=d.get("agrees_with_decision"),
        rationale=str(d.get("rationale", "")).strip(),
        recommended_action=str(d.get("recommended_action", "")).strip(),
        driver_instruction=str(d.get("driver_instruction", "")).strip(),
        model=getattr(backend, "model", "?"),
        used_images=bool(getattr(backend, "vision", False) and images),
        raw=raw,
    )


if __name__ == "__main__":
    # Offline smoke test: assemble a prompt and run the mock backend.
    ctx = {"risk_level": "BRAKE", "nearest_obstacle_m": "5.1 m", "ttc_s": "1.7 s",
           "ego_speed_kmh": 11,
           "detections": {"CAM_FRONT": ["car", "truck"], "CAM_FRONT_LEFT": ["person"]}}
    print(build_prompt(ctx, ["CAM_FRONT", "CAM_FRONT_LEFT", "BEV_LiDAR"]))
    print("-" * 60)
    res = analyze(MockVLM(), ctx, images=[b"x"], image_labels=["CAM_FRONT"])
    print(json.dumps(res.to_dict(), indent=2))


# --------------------------------------------------------------------------- #
# Scene narration (temporal, multi-keyframe) for the narrated-scene demo
# --------------------------------------------------------------------------- #
COMMENT_SCHEMA = (
    "Respond with ONLY this JSON object:\n"
    '{ "comment": "<one short spoken-style sentence describing what the ego vehicle '
    'sees right now and any hazard>" }\n'
    "Speak like a co-pilot narrating live. Use cues across ALL camera angles and the "
    "LiDAR BEV (depth/empty space). Max ~20 words."
)

SCENE_SUMMARY_SCHEMA = (
    "These images are sequential keyframes of ONE driving scene (surround cameras + LiDAR "
    "BEV). Respond with ONLY this JSON object:\n"
    "{\n"
    '  "summary": "<2-3 sentences: what happens across the scene, the setting, and the '
    'main risk>",\n'
    '  "hazards": ["<open-vocabulary hazard>", ...],\n'
    '  "scene_type": "<e.g. urban-intersection | parking | highway | residential>"\n'
    "}"
)


def build_comment_prompt(context):
    """Per-frame live co-pilot comment."""
    head = (SYSTEM_INSTRUCTION + "\n\nThis is one moment of a driving scene "
            "(6-camera surround montage + LiDAR BEV tile).\n")
    risk = context.get("risk_level", "unknown")
    head += f"Geometric risk now: {risk}"
    if context.get("nearest_obstacle_m") is not None:
        head += f"; nearest in-path obstacle {context['nearest_obstacle_m']}"
    if context.get("ttc_s") is not None:
        head += f"; TTC {context['ttc_s']}"
    return head + ".\n\n" + COMMENT_SCHEMA


def build_summary_prompt(context, n_keyframes):
    head = (SYSTEM_INSTRUCTION + f"\n\nYou are given {n_keyframes} keyframes spanning one "
            "scene in time order.\n")
    if context.get("risk_profile"):
        head += f"Per-keyframe geometric risk: {context['risk_profile']}.\n"
    return head + "\n" + SCENE_SUMMARY_SCHEMA


def comment_segment(backend, context, images):
    """One short live comment for a single moment (single composite image)."""
    raw = backend.generate(build_comment_prompt(context), images=images)
    d = _parse(raw)
    return str(d.get("comment", "")).strip() or "(no comment)"


def summarize_scene(backend, context, images):
    """Holistic narration over several keyframe images of one scene."""
    raw = backend.generate(build_summary_prompt(context, len(images or [])), images=images)
    d = _parse(raw)
    hazards = d.get("hazards", [])
    if isinstance(hazards, str):
        hazards = [hazards]
    return {
        "summary": str(d.get("summary", "")).strip(),
        "hazards": [str(h) for h in hazards],
        "scene_type": str(d.get("scene_type", "")).strip(),
    }


class MockNarrator(MockVLM):
    """Mock backend that returns plausible per-frame comments / scene summaries so the
    narrated-scene demo can be produced with no GPU or server."""

    _COMMENTS = [
        "Approaching a quiet urban street, parked cars line the right shoulder.",
        "A vehicle ahead sits close in our lane — easing off, watching the gap.",
        "Clear road opening up on the left; LiDAR shows free space ahead.",
        "Pedestrian near the right curb — holding speed, ready to slow.",
        "Intersection approaching, scanning cross-traffic on the side cameras.",
        "Path ahead is clear; maintaining a steady cruise.",
    ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self._i = 0

    def generate(self, prompt, images=None):
        if '"summary"' in prompt:
            return json.dumps({
                "summary": "The ego vehicle drives through a quiet urban street lined with "
                           "parked cars, slows for a nearby vehicle in-lane, then finds the "
                           "path ahead clearing. Main risk is the close in-lane vehicle.",
                "hazards": ["parked vehicles narrowing lane", "in-lane vehicle at close range",
                            "pedestrian near curb"],
                "scene_type": "urban-residential",
            })
        c = self._COMMENTS[self._i % len(self._COMMENTS)]
        self._i += 1
        return json.dumps({"comment": c})
