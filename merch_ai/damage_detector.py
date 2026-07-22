"""Damage-detection pipeline for the merchandise audit app.

Full Python port of the CURRENT detection logic in
nextjs-app/app/api/merchandise/analyze/route.ts, wrapped in a class so a
caller (e.g. a multi-image feature-matching/alignment pipeline that needs
damage detection run once per photo before merging overlapping detections
across photos of the same table) can invoke it directly, in-process, with no
network dependency on the Next.js app.

This is a port, not a wrapper: every stage below (LLM issue-judging prompts,
the detector-first CV sweep, localizer box refinement, post-localization
verification, IOU dedupe) mirrors route.ts line for line, using this server's
existing provider-agnostic LLM transport (app/llm.py) for the model calls.

One real dependency carries over unchanged from route.ts: the detector-first
sweep, box refinement, and segmentation still call the separate localizer
service (OmDet-Turbo + SlimSAM, see localizer/app.py) over HTTP via
LOCALIZER_URL. That is architectural in the original too (the CV detector runs
as its own service) — it is not something this port introduces. When that
service is unset/unreachable, this degrades to coarse LLM boxes, exactly like
route.ts does.

`python-server/app/routes_merchandise.py` is a separate, older Python port of
an earlier version of this route (missing the detector-first/localizer/
verification stages) — this module does not use or depend on it.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
from typing import Any, Optional

import httpx

from . import llm as llm_mod

# --------------------------------------------------------------------------- #
# Errors.
# --------------------------------------------------------------------------- #


class DamageDetectionError(Exception):
    """Raised when detection cannot complete. Carries the same metadata shape
    route.ts attaches to its error JSON responses (code/status/detail/
    provider/model/base_url), adapted to an exception since there is no HTTP
    request/response cycle here."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        status: Optional[int] = None,
        detail: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.detail = detail
        self.provider = provider
        self.model = model
        self.base_url = base_url


# --------------------------------------------------------------------------- #
# Bounding-box sanitizer — verbatim port of lib/merchandise/scoring.ts's
# sanitizeBbox: corner-pair/x1-x2 conversion, then a 0-100/0-1000 scale rescue
# before clamping, so a VLM's leaked coordinate convention doesn't collapse to
# a dropped zero-size box.
# --------------------------------------------------------------------------- #


def sanitize_bbox(bbox: Any) -> Optional[dict]:
    if not bbox:
        return None

    def num(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            v = float(value)
            if v != v or v in (float("inf"), float("-inf")):
                return None
            return v
        return None

    # Corner-pair array [x1, y1, x2, y2] — Qwen-VL's native bbox_2d format.
    if isinstance(bbox, (list, tuple)):
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = (num(v) for v in bbox)
        if None in (x1, y1, x2, y2):
            return None
        return sanitize_bbox(
            {"x": min(x1, x2), "y": min(y1, y2), "w": abs(x2 - x1), "h": abs(y2 - y1)}
        )

    if not isinstance(bbox, dict):
        return None

    # {x1, y1, x2, y2} objects: the same corner-pair convention under other keys.
    if bbox.get("x") is None and bbox.get("x1") is not None and bbox.get("x2") is not None:
        x1 = num(bbox.get("x1"))
        y1 = num(bbox.get("y1"))
        x2 = num(bbox.get("x2"))
        y2 = num(bbox.get("y2"))
        if None in (x1, y1, x2, y2):
            return None
        return sanitize_bbox(
            {"x": min(x1, x2), "y": min(y1, y2), "w": abs(x2 - x1), "h": abs(y2 - y1)}
        )

    x = num(bbox.get("x"))
    y = num(bbox.get("y"))
    w = num(bbox.get("w"))
    h = num(bbox.get("h"))
    if None in (x, y, w, h):
        return None

    # Rescue out-of-range coordinate conventions instead of clamp-deleting the
    # box: VLMs leak their trained coordinate space regardless of the prompt
    # (Qwen-VL emits 0-1000-scale values, others percentages). Values only
    # slightly past 1 are sloppy-but-normalized: clamp as before.
    extent = max(x, y, x + w, y + h)
    if extent > 1.5:
        scale = 100 if extent <= 100 else (1000 if extent <= 1000 else None)
        if scale is None:
            return None  # pixel coords of an unknown image size
        x /= scale
        y /= scale
        w /= scale
        h /= scale

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))

    if w <= 0.005 or h <= 0.005:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _first_defined(*values: Any) -> Any:
    """JS `a ?? b ?? c` — first value that is not None (JS null/undefined)."""
    for v in values:
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------- #
# Box geometry.
# --------------------------------------------------------------------------- #


def normalize_polygon(raw: Any) -> Optional[list]:
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    points = []
    for point in raw:
        if not isinstance(point, dict):
            continue
        px, py = point.get("x"), point.get("y")
        if isinstance(px, bool) or isinstance(py, bool):
            continue
        if not isinstance(px, (int, float)) or not isinstance(py, (int, float)):
            continue
        px, py = float(px), float(py)
        if px != px or py != py:
            continue
        points.append({"x": max(0.0, min(1.0, px)), "y": max(0.0, min(1.0, py))})
    return points if len(points) >= 3 else None


def iou_boxes(a: dict, b: dict) -> float:
    """Intersection-over-union of two normalized boxes (0 = disjoint, 1 = identical)."""
    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
    iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def map_crop_box_to_image(crop_rel: Any, region: dict) -> Optional[dict]:
    """Map a crop-relative box (0-1 within the crop) back to full-image
    normalized space using the crop's region. None if the box is unusable."""
    rel = sanitize_bbox(crop_rel)
    if not rel:
        return None
    return sanitize_bbox(
        {
            "x": region["x"] + rel["x"] * region["w"],
            "y": region["y"] + rel["y"] * region["h"],
            "w": rel["w"] * region["w"],
            "h": rel["h"] * region["h"],
        }
    )


# A scene-pass finding whose box overlaps a detector-first finding of the same
# type at least this much is the same physical problem — keep the
# detector-first one (its box came from the detector, its verdict from a
# dedicated crop review).
DETECTOR_DEDUPE_IOU = 0.45


def _default_correction_max_iou() -> float:
    raw = os.getenv("LOCALIZER_CORRECTION_MAX_IOU")
    try:
        v = float(raw) if raw else None
    except ValueError:
        v = None
    return v if v is not None and 0 < v <= 1 else 0.5


# The detector's box is usually right; a VLM "correction" that only nudges it
# (high IoU with the current box) risks degrading a good box for no gain. Only
# accept a correction that SUBSTANTIALLY disagrees. Tunable via
# LOCALIZER_CORRECTION_MAX_IOU.
DEFAULT_CORRECTION_MAX_IOU = _default_correction_max_iou()


# --------------------------------------------------------------------------- #
# JSON recovery ladder (verbatim port of the analyze route helpers).
# --------------------------------------------------------------------------- #

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"```\s*$", re.IGNORECASE)
_FENCE_ANY = re.compile(r"```(?:json)?", re.IGNORECASE)


def _strip_json_fence(text: str) -> str:
    trimmed = text.strip()
    unfenced = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", trimmed)).strip()
    start = unfenced.find("[")
    end = unfenced.rfind("]")
    if start >= 0 and end > start:
        return unfenced[start : end + 1]
    return unfenced


def _extract_balanced_objects(text: str) -> list:
    objects: list = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and ("caption" in parsed or "detected_issues" in parsed):
                        objects.append(parsed)
                except Exception:
                    pass  # a later object may still be recoverable
                start = -1
    return objects


def parse_model_json(content: str) -> list:
    """Tolerant parse of a JSON array response (the base + phone-state + crop
    verification passes all return an array)."""
    cleaned = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", content.strip())).strip()
    attempts = [cleaned, _strip_json_fence(content), _FENCE_ANY.sub("", cleaned).replace("```", "").strip()]
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            return parsed["results"]
        if isinstance(parsed, dict) and isinstance(parsed.get("images"), list):
            return parsed["images"]
    balanced = _extract_balanced_objects(cleaned)
    if balanced:
        return balanced
    raise ValueError("No recoverable JSON array or result objects found.")


def parse_merged_object(content: str) -> Optional[dict]:
    """Tolerant parse of the merged set-of-marks call's single JSON object."""
    cleaned = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", content.strip())).strip()
    attempts = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        attempts.append(cleaned[start : end + 1])
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _clean_str(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def normalize_vision_results(raw: Any, request_images: list) -> list:
    """Normalize the base (non-merged) pass's JSON array into VisionResult dicts."""
    arr = raw if isinstance(raw, list) else []
    results = []
    for index, image in enumerate(request_images):
        item = arr[index] if index < len(arr) and isinstance(arr[index], dict) else {}
        detected = item.get("detected_issues") if isinstance(item.get("detected_issues"), list) else []
        issues = []
        for issue in detected:
            if not isinstance(issue, dict):
                continue
            issue_type_id = str(_first_defined(issue.get("issue_type_id"), "") or "")
            if not issue_type_id:
                continue
            confidence_num = _finite_number(issue.get("confidence"))
            issues.append(
                {
                    "issue_type_id": issue_type_id,
                    "confidence": max(0.0, min(1.0, confidence_num)) if confidence_num is not None else 0.7,
                    "evidence": issue.get("evidence") if isinstance(issue.get("evidence"), str) else "",
                    "object_label": _clean_str(issue.get("object_label")),
                    # Fall back to Qwen-VL's native bbox_2d key (and generic box)
                    # when the model ignores the requested shape; sanitize_bbox
                    # handles the corner-pair/0-1000 conversions.
                    "bbox": sanitize_bbox(
                        _first_defined(issue.get("bbox"), issue.get("bbox_2d"), issue.get("box"))
                    ),
                    "polygon": None,
                }
            )
        results.append(
            {
                "image_id": image.get("id"),
                "caption": item.get("caption") if isinstance(item.get("caption"), str) else "",
                "detected_issues": issues,
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Reference photos + prompt builders (prompt text reproduced verbatim from
# route.ts / the BBOX_RULES + merged-prompt additions it has now).
# --------------------------------------------------------------------------- #


def collect_reference_items(fixture_config: dict) -> list:
    """Collect every brand-supplied reference photo across the taxonomy,
    flattened in taxonomy order."""
    items = []
    for issue in fixture_config.get("issue_taxonomy") or []:
        for ref in issue.get("reference_images") or []:
            if not ref or not ref.get("url"):
                continue
            items.append(
                {
                    "issue_type_id": issue.get("id"),
                    "label": issue.get("label"),
                    "caption": ref.get("caption") or "",
                    "data_url": ref["url"],
                    "box": sanitize_bbox(ref.get("box")),
                }
            )
    return items


def collect_detector_labels(fixture_config: dict) -> list:
    """Plain visual nouns for the detector-first sweep, deduped, in first-seen order."""
    labels: list = []
    seen: set = set()
    for issue in fixture_config.get("issue_taxonomy") or []:
        for label in issue.get("object_labels") or []:
            trimmed = (label or "").strip()
            if trimmed and trimmed not in seen:
                seen.add(trimmed)
                labels.append(trimmed)
    return labels


def _pct(v: float) -> str:
    # JS Math.round (half away from zero for positives), not banker's rounding.
    return f"{math.floor(v * 100 + 0.5)}%"


def format_ref_box(box: Optional[dict]) -> Optional[str]:
    if not box:
        return None
    return (
        f"top-left ({_pct(box['x'])}, {_pct(box['y'])}), "
        f"size {_pct(box['w'])} wide × {_pct(box['h'])} tall of the reference image"
    )


def _find_fixture_type(fixture_config: dict, fixture_type_id: Optional[str]) -> Optional[dict]:
    for item in fixture_config.get("fixture_types") or []:
        if item.get("id") == fixture_type_id:
            return item
    return None


# Precise, unambiguous box convention shared by every prompt that emits a bbox.
BBOX_RULES = """BOUNDING BOX RULES (follow EXACTLY — boxes are usually wrong when these are ignored):
- Coordinate space: the image's TOP-LEFT corner is (0,0), its BOTTOM-RIGHT corner is (1,1). All values are fractions of the FULL image.
- "x","y" are the position of the box's TOP-LEFT corner. "w","h" are the box's width and height. Do NOT output the center point, and do NOT output the bottom-right corner.
- Measure against the WHOLE image's width and height — never relative to a shelf, product, or any sub-region. Respect the photo's true aspect ratio: in a wide (landscape) photo an object usually spans a smaller "w" than "h", and vice-versa.
- The box must TIGHTLY enclose ONLY the one object named in object_label — exclude neighbouring products, the shelf, signage, and empty background. Aim for the object to fill ~90% of the box.
- Emit ONE box per physical object. If two objects share the same issue (e.g. two off phones), return two separate detections, each with its own tight box.
- Stay in bounds: 0 ≤ x, 0 ≤ y, x + w ≤ 1, y + h ≤ 1. Never emit negative values or values above 1.
- A downstream detector refines your box, but it can only refine a box that is already on the CORRECT object and roughly the right size — correct object identity and rough tightness matter more than pixel precision."""


def _taxonomy_for_prompt(fixture_config: dict, *, with_reference_flag: bool) -> list:
    taxonomy = []
    for item in fixture_config.get("issue_taxonomy") or []:
        signals = item.get("detection_signals") or {}
        entry = {
            "id": item.get("id"),
            "label": item.get("label"),
            "expected": item.get("expected_state"),
            "category": item.get("category"),
            "failure_modes": item.get("failure_modes"),
            "image_cues": signals.get("image_cues"),
        }
        if with_reference_flag:
            entry["has_reference_images"] = len(item.get("reference_images") or []) > 0
        taxonomy.append(entry)
    return taxonomy


def build_merged_system_prompt(
    fixture_config: dict,
    fixture_type_id: Optional[str],
    observations: Optional[str],
    candidates: list,
    reference_items: list,
    has_grid: bool,
) -> str:
    fixture_type = _find_fixture_type(fixture_config, fixture_type_id)
    taxonomy = _taxonomy_for_prompt(fixture_config, with_reference_flag=False)

    if reference_items:
        ref_lines = []
        for i, ref in enumerate(reference_items):
            box_text = format_ref_box(ref["box"])
            line = f"  • Image {i + 1}: example of '{ref['issue_type_id']}' ({ref['label']})"
            if ref["caption"]:
                line += f' — "{ref["caption"]}"'
            if box_text:
                line += f"; failure pattern at {box_text}"
            ref_lines.append(line)
        refs_block = (
            f"\nThe FIRST {len(reference_items)} image(s) are brand-supplied REFERENCE examples — training "
            "material only. NEVER flag issues in them and NEVER describe them in your output:\n"
            + "\n".join(ref_lines)
            + "\nA reference shows ONE manifestation; flag every instance that fits an issue's taxonomy scope "
            "even when it looks nothing like the reference, and never skip an issue type just because it has "
            "no reference.\n"
        )
    else:
        refs_block = ""

    marks_list = "\n".join(
        f'  {i + 1}: "{c["label"]}" (detector confidence {c["score"]:.2f})' for i, c in enumerate(candidates)
    )
    audit_intro = "After the reference image(s), the" if reference_items else "The"

    if candidates:
        view_b_note = (
            "\n- View B: a grid of zoomed crops of those same numbered boxes, for close detail. Use View B to "
            "inspect each object and View A to compare it against its neighbours (e.g. a dark screen next to "
            "lit ones)."
            if has_grid
            else ""
        )
        views_block = (
            f"{audit_intro} AUDIT photo follows in {'TWO views' if has_grid else 'ONE view'}:\n"
            f"- View A: the photo with {len(candidates)} numbered RED box(es); each marks ONE object an object "
            f"detector found.{view_b_note}\n"
            f"Marked boxes (mark number: detector label):\n{marks_list}"
        )
    else:
        views_block = (
            f'{audit_intro} AUDIT photo follows. No detector marks are available for it — return "marks": {{}} '
            'and report every finding in "scene_issues".'
        )

    fixture_name = fixture_type.get("name") if fixture_type else "unspecified fixture"
    fixture_description = fixture_type.get("description") if fixture_type else "none"
    obs = (observations or "").strip() or "none"

    return f"""You are a local AI retail fixture auditor analyzing ONE audit photo for merchandise, planogram, stock, demo, pricing, safety, and brand-compliance issues.

{views_block}
{refs_block}
Fixture type: {fixture_name}
Fixture description: {fixture_description}
Field observations: {obs}

Return ONE raw JSON object — no markdown fences, no prose — shaped exactly:
{{
  "caption": "<one-sentence photo caption>",
  "marks": {{ "<mark number>": {{ "verdict": "ok" | "not_object" | "bad_box", "screen": "lit" | "dark" | "no_screen", "issues": [ {{ "id": "<taxonomy id>", "conf": 0.0-1.0, "ev": "<short specific visual reason>" }} ] }} }},
  "scene_issues": [ {{ "id": "<taxonomy id>", "conf": 0.0-1.0, "ev": "<specific evidence>", "object_label": "<1-3 word plain noun>", "bbox": {{ "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4 }} }} ]
}}

TASK 1 — judge EVERY numbered mark independently (every mark number MUST appear as a key in "marks"; a row of similar objects commonly mixes good and bad units):
- "verdict": "ok" = the box frames a REAL, PHYSICAL instance of its listed label on the fixture. "not_object" = it frames a PRINTED or RENDERED image of one (poster, box art, on-screen content), signage, a reflection, or a different object. "bad_box" = right object but the box badly mis-frames it (cut in half, mostly background).
- "screen": for devices with screens, judge by EMITTED LIGHT ONLY, ignoring content: "lit" = emits ANY light of its own (plain white page, bare logo, wallpaper, gradient, any glow); "dark" = black/near-black glass emitting none (at most reflecting the room); "no_screen" = no screen or not visible.
- "issues": every taxonomy issue whose FAILURE state is visible on that object; [] when it looks fine. Most objects are fine. Powered-off means "screen": "dark" — a lit screen is NEVER a power issue, whatever it shows.

TASK 2 — "scene_issues": every OTHER taxonomy issue visible in the photo that no numbered mark covers (empty slots, missing/wrong pricing, missing or damaged signage, obstructions, safety hazards, cleanliness, competitor items, ...). Do NOT repeat a finding already reported under a mark. object_label must be a plain, visually-groundable noun — it is fed verbatim to an object detector. Boxes here are refined downstream, so rough-but-on-the-right-object beats precise-but-wrong.

{BBOX_RULES}

Issue taxonomy:
{json.dumps(taxonomy, separators=(",", ":"), ensure_ascii=False)}"""


def build_system_prompt(
    fixture_config: dict,
    fixture_type_id: Optional[str],
    observations: Optional[str],
    images: list,
    reference_items: list,
) -> str:
    fixture_type = _find_fixture_type(fixture_config, fixture_type_id)
    taxonomy = _taxonomy_for_prompt(fixture_config, with_reference_flag=True)

    if reference_items:
        ref_lines = []
        for i, ref in enumerate(reference_items):
            box_text = format_ref_box(ref["box"])
            line = f"  • Image {i + 1}: example of '{ref['issue_type_id']}' ({ref['label']})"
            if ref["caption"]:
                line += f' — "{ref["caption"]}"'
            if box_text:
                line += f"; the failure pattern is at {box_text} — match that object, not the surrounding scene"
            ref_lines.append(line)
        refs_block = (
            "\n\nREFERENCE PHOTOS — how they fit in:\n"
            f"The FIRST {len(reference_items)} image(s) in this request are brand-supplied REFERENCE examples. "
            "They are training material: NEVER flag issues in them and NEVER describe their contents in your "
            "output. Each shows ONE example of how an issue tends to look on this brand's fixtures:\n"
            + "\n".join(ref_lines)
            + f"\nThe remaining {len(images)} image(s) are the AUDIT photos to analyze, in order.\n\n"
            "Use references as ILLUSTRATIONS, not exclusive definitions:\n"
            "- A reference shows ONE manifestation; the same issue type can appear in other visual forms. The "
            "taxonomy label, expected_state, and failure_modes describe the full SCOPE — flag every instance "
            "that fits the scope even when it looks nothing like the reference object.\n"
            "- References do NOT suppress other issue types. An issue with no reference photo is detected "
            "purely from its taxonomy text — its lack of a reference is NOT a reason to skip it."
        )
    else:
        refs_block = ""

    fixture_name = fixture_type.get("name") if fixture_type else "unspecified fixture"
    fixture_description = fixture_type.get("description") if fixture_type else "none"
    obs = (observations or "").strip() or "none"
    not_reference = " (NOT the reference photos)" if reference_items else ""

    return f"""You are a local AI retail fixture auditor. Analyze {len(images)} audit photo(s) for merchandise, planogram, stock, demo, pricing, safety, and brand-compliance issues.

DETECTION CRITERIA (in priority order):
For each issue type in the taxonomy, evaluate each audit photo against:
  (1) the issue's label, expected_state, failure_modes, and image_cues — these are the PRIMARY criteria; they describe the full scope of what counts as that issue.
  (2) any reference photo provided for that issue — a SECONDARY illustration of one form the issue can take, helpful as a positive example but never required for a match.
A visible element matches an issue if it fits criterion (1), regardless of whether it also resembles criterion (2). The same issue type can manifest in multiple forms within one photo and across photos — flag every distinct instance.

Fixture type: {fixture_name}
Fixture description: {fixture_description}
Field observations: {obs}

For EACH audit photo, do BOTH:
1. Write a one-sentence caption.
2. List EVERY issue with visible evidence. Multiple distinct issues per photo are common — return ALL of them as separate detections. Do NOT collapse unrelated issues into one. Do NOT skip an issue type just because no reference photo was provided for it.

DETECTION GUIDANCE FOR COMMON CASES (apply when the relevant taxonomy entries exist):
- "Demo phone powered off / screen off / black screen": a handset counts as OFF when its screen is black, blank, or near-black with no visible UI, lockscreen, wallpaper, app icon, or demo loop. A handset is ON if its screen shows ANY illuminated content (even a dim wallpaper or logo). Inspect each handset individually — a row of demo phones commonly mixes ON and OFF units, and you must flag the OFF ones even when neighbouring phones are clearly lit. When borderline (could be off, or just a dark wallpaper), prefer to flag — a rep re-checking is cheaper than missing a real off-state. A black handset in the front-center of a Samsung/Galaxy display is a powered-off demo unit.
- Distinguish phone-shaped handheld demo units and tablets sitting on the fixture from large TVs, wall monitors, vertical brand signs, dark furniture, and black price labels — only flag the handsets/tablets.
- Multiple instances of the same issue type in one photo (e.g. two separate off phones) are SEPARATE detections, each with its own bbox.{refs_block}

Rules:
- Return raw JSON only. No markdown fences. No ```json. No prose.
- Return exactly one JSON array. Do not close the array early and then add more objects after it.
- The JSON array must contain exactly {len(images)} entries in the same order as the AUDIT photos{not_reference}.
- evidence: name the specific object you saw (be concrete, e.g. "colorful portable speaker on the left of the demo table" or "stack of product boxes on the floor in front of the fixture").
- object_label: a SHORT 1-3 word noun phrase naming just the physical object as a downstream OBJECT DETECTOR would recognise it — a concrete, common, visually-groundable noun (e.g. "portable speaker", "product boxes", "demo phone"). No location words, no state adjectives, no brand/model names, no issue words. This label is fed verbatim to the detector, so a wrong or vague label produces a wrong box.
- bbox: a tight normalized box around the object named in object_label, in the AUDIT image, per the BOUNDING BOX RULES below.
- Use only issue_type_id values that exist in the taxonomy.

{BBOX_RULES}

Output shape:
[
  {{
    "caption": "short caption",
    "detected_issues": [
      {{
        "issue_type_id": "taxonomy_id",
        "confidence": 0.0,
        "evidence": "specific visual evidence",
        "object_label": "short object name",
        "bbox": {{ "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4 }}
      }}
    ]
  }}
]

Issue taxonomy:
{json.dumps(taxonomy, separators=(",", ":"), ensure_ascii=False)}"""


def build_phone_state_verification_prompt(
    fixture_config: dict, fixture_type_id: Optional[str], observations: Optional[str], images: list
) -> str:
    fixture_type = _find_fixture_type(fixture_config, fixture_type_id)
    fixture_name = fixture_type.get("name") if fixture_type else "unspecified fixture"
    obs = (observations or "").strip() or "none"

    return f"""You are a strict retail demo-phone power-state auditor.

Your only task is to inspect smartphone/demo handset screens in {len(images)} fixture audit photo(s).

Fixture type: {fixture_name}
Field observations: {obs}

Required behavior:
- Examine every phone-shaped handset or tablet on the fixture/table/pedestal, one by one.
- Compare each handset screen against neighboring lit phones.
- If a handset screen is black, blank, unlit, or near-black with no UI/lockscreen/wallpaper/app/demo content, flag it as "demo_unit_power_off".
- Do not skip a black phone just because other phones in the same display are powered on.
- Do not flag large wall TVs, signage, brand pillars, dark furniture, black price labels, or display monitors as phones.
- When a black phone is visible in the front center of a Samsung/Galaxy fixture, it is a powered-off demo unit and must be flagged.
- object_label must stay a plain detector-friendly noun ("demo phone" or "tablet"); the bbox must tightly enclose just that one handset.

{BBOX_RULES}

Return raw JSON only. No markdown fences. No ```json. No prose. Return exactly one JSON array with exactly {len(images)} entries in the same order as the photos.

Output shape:
[
  {{
    "caption": "brief phone power-state caption",
    "detected_issues": [
      {{
        "issue_type_id": "demo_unit_power_off",
        "confidence": 0.0,
        "evidence": "specific phone location and why it is off",
        "object_label": "demo phone",
        "bbox": {{ "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4 }}
      }}
    ]
  }}
]

If no powered-off phone-shaped demo unit is visible in a photo, use "detected_issues": []."""


# --------------------------------------------------------------------------- #
# LLM vision call + phone-state merge.
# --------------------------------------------------------------------------- #


async def _call_llm_vision(
    llm: dict,
    system: str,
    image_data_urls: list,
    user_text: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 3500,
) -> str:
    """`image_data_urls` are full data URLs ordered exactly as the prompt
    expects: reference photos first, audit photos after."""
    content = [{"type": "text", "text": user_text}] + [
        {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls
    ]
    return await llm_mod.chat_completion(
        provider=llm["provider"],
        model=llm["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _parse_vision_results(content: str, images: list) -> list:
    return normalize_vision_results(parse_model_json(content), images)


def _has_powered_off_phone(results: list) -> bool:
    return any(
        issue.get("issue_type_id") == "demo_unit_power_off"
        for result in results
        for issue in result.get("detected_issues", [])
    )


def _merge_phone_verification_results(primary: list, verification: list) -> list:
    primary_by_image = {result["image_id"]: result for result in primary}
    merged = []
    for result in primary:
        verified = next((item for item in verification if item.get("image_id") == result["image_id"]), None)
        if not verified:
            merged.append(result)
            continue
        phone_findings = [
            issue for issue in verified.get("detected_issues", []) if issue.get("issue_type_id") == "demo_unit_power_off"
        ]
        if not phone_findings:
            merged.append(result)
            continue
        primary_result = primary_by_image.get(result["image_id"])
        already_has_phone_finding = bool(primary_result) and any(
            issue.get("issue_type_id") == "demo_unit_power_off" for issue in primary_result.get("detected_issues", [])
        )
        merged.append(
            {
                **result,
                "caption": result.get("caption") or verified.get("caption"),
                "detected_issues": result["detected_issues"]
                if already_has_phone_finding
                else [*result["detected_issues"], *phone_findings],
            }
        )
    return merged


# --------------------------------------------------------------------------- #
# Detector-first CV sweep (talks to the localizer service: /detect, /segment).
# --------------------------------------------------------------------------- #


async def _detect_sweep(base: str, image_data_url: str, labels: list, timeout_ms: float) -> dict:
    """POST /detect — every candidate box for `labels`, plus the set-of-marks
    views (annotated photo + numbered crop grid) for the single-call judge."""
    timeout = httpx.Timeout(timeout_ms / 1000.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post(
                f"{base}/detect",
                json={
                    "image_data_url": image_data_url,
                    "labels": labels,
                    "return_annotated": True,
                    "return_grid": True,
                },
            )
        if res.status_code >= 400:
            return {"candidates": [], "annotated": None, "grid": None}
        payload = res.json()
        candidates = []
        for index, cand in enumerate(payload.get("candidates") or []):
            score = cand.get("score")
            candidates.append(
                {
                    "id": index,
                    "label": (str(cand.get("label") or "").strip()) or "object",
                    "score": score if isinstance(score, (int, float)) and not isinstance(score, bool) else 0,
                    "box": sanitize_bbox(cand.get("box")),
                }
            )
        annotated = payload.get("annotated") if isinstance(payload.get("annotated"), str) and payload.get("annotated") else None
        grid = payload.get("grid") if isinstance(payload.get("grid"), str) and payload.get("grid") else None
        return {"candidates": candidates, "annotated": annotated, "grid": grid}
    except Exception as detect_error:
        print(f"[merchandise] detector sweep unavailable: {detect_error}")
        return {"candidates": [], "annotated": None, "grid": None}


async def _segment_boxes(base: str, image_data_url: str, boxes: list, timeout_ms: float) -> list:
    """POST /segment — polygons for confirmed boxes, aligned by index."""
    if not boxes:
        return []
    timeout = httpx.Timeout(timeout_ms / 1000.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post(f"{base}/segment", json={"image_data_url": image_data_url, "boxes": boxes})
        if res.status_code >= 400:
            return [None for _ in boxes]
        payload = res.json()
        polygons = payload.get("polygons") or []
        return [normalize_polygon(polygons[i]) if i < len(polygons) else None for i in range(len(boxes))]
    except Exception:
        return [None for _ in boxes]


async def _analyze_image_merged(
    llm: dict,
    fixture_config: dict,
    fixture_type_id: Optional[str],
    observations: Optional[str],
    image: dict,
    reference_items: list,
    labels: list,
    base: str,
    timeout_ms: float,
) -> dict:
    """One merged set-of-marks call for one audit image: detect marks, judge
    them all AND collect scene-level issues in a single LLM request."""
    sweep = await _detect_sweep(base, image["data_url"], labels, timeout_ms)
    candidates = sweep["candidates"]
    system = build_merged_system_prompt(
        fixture_config, fixture_type_id, observations, candidates, reference_items, bool(sweep["grid"])
    )

    image_urls = (
        [ref["data_url"] for ref in reference_items]
        + [sweep["annotated"] or image["data_url"]]
        + ([sweep["grid"]] if sweep["grid"] else [])
    )
    user_text = (
        f"Audit this photo: judge all {len(candidates)} numbered mark(s) and report any other visible issues."
        if candidates
        else "Audit this photo and report every visible issue."
    )

    raw = await _call_llm_vision(
        llm, system, image_urls, user_text, max_tokens=min(6000, 2200 + 100 * len(candidates))
    )
    parsed = parse_merged_object(raw)
    if parsed is None:
        raise DamageDetectionError(f"The LLM did not return valid JSON for {image['id']}. Raw: {raw[:300]}")

    valid_ids = {item.get("id") for item in fixture_config.get("issue_taxonomy") or []}
    detected: list = []
    cleared: list = []
    rejected = 0
    bad_box = 0

    def read_issue(item: Any, fallback_evidence: str) -> Optional[dict]:
        if not isinstance(item, dict):
            return None
        issue_id = str(_first_defined(item.get("id"), item.get("issue_type_id"), "") or "")
        if issue_id not in valid_ids:
            return None
        conf_raw = _first_defined(item.get("conf"), item.get("confidence"))
        conf = (
            max(0.0, min(1.0, conf_raw)) if isinstance(conf_raw, (int, float)) and not isinstance(conf_raw, bool) else 0.7
        )
        ev_raw = _first_defined(item.get("ev"), item.get("evidence"))
        ev = ev_raw.strip() if isinstance(ev_raw, str) and ev_raw.strip() else fallback_evidence
        return {"id": issue_id, "conf": conf, "ev": ev}

    # Marks come back as a keyed map ("1": {...}); tolerate an array too.
    marks_raw = parsed.get("marks")
    mark_entries: list = []
    if isinstance(marks_raw, list):
        for index, entry in enumerate(marks_raw):
            if isinstance(entry, dict):
                mark_num = _first_defined(entry.get("id"), entry.get("mark"), index + 1)
                try:
                    mark_entries.append((int(mark_num), entry))
                except (TypeError, ValueError):
                    pass
    elif isinstance(marks_raw, dict):
        for key, value in marks_raw.items():
            if isinstance(value, dict):
                try:
                    mark_entries.append((int(key), value))
                except ValueError:
                    pass

    judged: set = set()
    for mark, record in mark_entries:
        cand = candidates[mark - 1] if 0 < mark <= len(candidates) else None
        if not cand or not cand.get("box") or mark in judged:
            continue
        judged.add(mark)

        verdict = str(_first_defined(record.get("verdict"), record.get("object"), "") or "").lower()
        if "not" in verdict:
            rejected += 1
            continue
        if "bad" in verdict:
            bad_box += 1
            continue

        # Structural gate: the model classifies emitted light separately from
        # issues, and code enforces the rule it tends to break in prose — a LIT
        # screen (any glow) cannot be powered off.
        screen_state = str(_first_defined(record.get("screen"), record.get("screen_state"), "") or "").lower()

        kept = []
        for item in record.get("issues") if isinstance(record.get("issues"), list) else []:
            issue = read_issue(item, f"Found on marked {cand['label']} #{mark}")
            if not issue:
                continue
            if issue["id"] == "demo_unit_power_off" and screen_state == "lit":
                continue
            kept.append(
                {
                    "issue_type_id": issue["id"],
                    "confidence": issue["conf"],
                    "evidence": issue["ev"],
                    "object_label": cand["label"],
                    "bbox": cand["box"],
                    "polygon": None,
                    "verified": True,
                    "verification_note": f"detector-first: judged as mark #{mark}",
                    "origin": "detector_first",
                }
            )
        if kept:
            detected.extend(kept)
        else:
            cleared.append({"object_label": cand["label"], "box": cand["box"], "score": cand["score"]})

    for item in parsed.get("scene_issues") if isinstance(parsed.get("scene_issues"), list) else []:
        issue = read_issue(item, "")
        if not issue or not issue["ev"]:
            continue
        detected.append(
            {
                "issue_type_id": issue["id"],
                "confidence": issue["conf"],
                "evidence": issue["ev"],
                "object_label": _clean_str(item.get("object_label")),
                "bbox": sanitize_bbox(_first_defined(item.get("bbox"), item.get("bbox_2d"), item.get("box"))),
                "polygon": None,
                "origin": "scene",
            }
        )

    # Mark findings keep the detector's box and skip re-localization; give them
    # their polygon here instead.
    mark_issues = [i for i in detected if i.get("origin") == "detector_first" and i.get("bbox")]
    if mark_issues:
        polygons = await _segment_boxes(base, image["data_url"], [i["bbox"] for i in mark_issues], timeout_ms)
        for issue, polygon in zip(mark_issues, polygons):
            issue["polygon"] = polygon

    result = {
        "image_id": image["id"],
        "caption": parsed.get("caption") if isinstance(parsed.get("caption"), str) else "",
        "detected_issues": detected,
    }
    if cleared:
        result["cleared_objects"] = cleared

    stats = {
        "candidates": len(candidates),
        "issues": len(mark_issues),
        "cleared": len(cleared),
        "rejected": rejected,
        "bad_box": bad_box,
    }
    return {"result": result, "stats": stats}


# --------------------------------------------------------------------------- #
# Localizer box refinement (talks to the localizer service: /localize) +
# post-localization LLM verification.
# --------------------------------------------------------------------------- #


async def _localize_vision_results(
    results: list, images: list, want_crops: bool, base: str, timeout_ms: float
) -> dict:
    """Refine each detection's coarse LLM box into a precise box + polygon
    using the localizer service. Best-effort: unreachable/unset -> LLM boxes
    kept unchanged. Returns crops keyed by `id(issue)` (Python has no
    reference-keyed dict; this stands in for the TS `Map<DetectedIssue, ...>`)."""
    crop_by_issue: dict = {}
    total = sum(
        1 for r in results for issue in r.get("detected_issues", []) if issue.get("origin") != "detector_first"
    )
    if not base:
        return {"crops": crop_by_issue, "configured": False, "reachable": False, "refined": 0, "total": total}

    data_url_by_id = {img["id"]: img["data_url"] for img in images}
    timeout = httpx.Timeout(timeout_ms / 1000.0)
    counters = {"ok": 0, "failed": 0, "refined": 0}

    async def process_result(result: dict) -> None:
        data_url = data_url_by_id.get(result["image_id"])
        if not data_url or not result.get("detected_issues"):
            return

        entries = []
        for index, issue in enumerate(result["detected_issues"]):
            if issue.get("origin") == "detector_first":
                continue
            label = (issue.get("object_label") or issue.get("evidence") or "")[:80]
            entries.append({"id": str(index), "label": label, "box": issue.get("bbox")})
        if not entries or all(not e["label"].strip() for e in entries):
            return

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(
                    f"{base}/localize",
                    json={"image_data_url": data_url, "items": entries, "return_crops": want_crops},
                )
            if res.status_code >= 400:
                counters["failed"] += 1
                return
            counters["ok"] += 1
            payload = res.json()
            for entry in payload.get("results") or []:
                try:
                    index = int(entry.get("id"))
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= len(result["detected_issues"]):
                    continue
                issue = result["detected_issues"][index]
                # Only overwrite the box when the DETECTOR actually located it;
                # a fallback entry echoes the LLM box, which is already in place.
                if entry.get("source") == "detector":
                    refined_box = sanitize_bbox(entry.get("box"))
                    if refined_box:
                        issue["bbox"] = refined_box
                        counters["refined"] += 1
                polygon = normalize_polygon(entry.get("polygon"))
                if polygon:
                    issue["polygon"] = polygon
                if want_crops and isinstance(entry.get("crop"), str) and entry.get("crop"):
                    region = sanitize_bbox(entry.get("crop_box"))
                    if region:
                        crop_by_issue[id(issue)] = {"issue": issue, "crop": entry["crop"], "region": region}
        except Exception as localize_error:
            counters["failed"] += 1
            print(f"[merchandise] localizer unavailable, using LLM boxes: {localize_error}")

    await asyncio.gather(*[process_result(r) for r in results])

    # Reachable if any request succeeded; if we had nothing to send, treat the
    # localizer as reachable (it simply wasn't needed).
    reachable = counters["ok"] > 0 or (counters["ok"] == 0 and counters["failed"] == 0)
    if not reachable:
        print(
            f"[merchandise] localizer configured ({base}) but unreachable — "
            f"returning coarse LLM boxes for all {total} detection(s)."
        )
    return {
        "crops": crop_by_issue,
        "configured": True,
        "reachable": reachable,
        "refined": counters["refined"],
        "total": total,
    }


async def _verify_detections(
    llm: dict, results: list, crop_by_issue: dict, fixture_config: dict, correction_max_iou: float
) -> None:
    """Post-localization verification: for each detection that has a tight
    crop, ask the LLM whether the crop genuinely shows the claimed object and
    issue, and — when the box is loose or off — return a corrected box.
    Rejected detections are KEPT but marked verified=False rather than
    silently dropped. Best-effort: any failure leaves detections unverified."""
    taxonomy = {item.get("id"): item for item in fixture_config.get("issue_taxonomy") or []}

    async def process_result(result: dict) -> None:
        entries = []
        for index, issue in enumerate(result.get("detected_issues", [])):
            ctx = crop_by_issue.get(id(issue))
            if ctx:
                entries.append({"issue": issue, "index": index, "ctx": ctx})
        if not entries:
            return

        lines = []
        for k, e in enumerate(entries):
            issue = e["issue"]
            tax = taxonomy.get(issue["issue_type_id"])
            label = tax.get("label") if tax else issue["issue_type_id"]
            expected = f" Expected GOOD state: {tax.get('expected_state')}." if tax and tax.get("expected_state") else ""
            lines.append(
                f'Detection {k}: object claimed = "{issue.get("object_label") or ""}", issue claimed = "{label}".'
                f'{expected} Evidence claimed: "{issue.get("evidence")}". (crop #{k + 1} below)'
            )

        system = (
            "You are a strict verifier for a retail fixture auditor. Each numbered CROP below is the exact "
            "image region an upstream model flagged for an issue. For EACH detection do TWO things:\n"
            "1. VERDICT — decide whether the crop genuinely shows the claimed object AND the claimed issue. "
            'Mark "false_positive" when the crop shows a DIFFERENT object than claimed, shows the object in '
            'its GOOD/expected state (i.e. no real issue), or is too ambiguous/empty to justify flagging. '
            'Otherwise mark "confirmed".\n'
            "2. BOX (optional) — if the claimed object is NOT tightly framed by the crop (it's off-center, "
            "only partly shown, or the crop is mostly background), return a corrected box that tightly "
            "encloses just the object, in coordinates RELATIVE TO THIS CROP: the crop's TOP-LEFT corner is "
            '(0,0) and its BOTTOM-RIGHT is (1,1). "x","y" are the box\'s TOP-LEFT corner (NOT its center); '
            '"w","h" are its width and height as fractions of the crop. Stay in bounds (x+w ≤ 1, y+h ≤ 1). '
            'OMIT "box" entirely when the crop already frames the object tightly. Never return a box for a '
            "false_positive.\n"
            f"Return raw JSON only — no prose, no markdown fences: an array with EXACTLY {len(entries)} objects, "
            "in the same order as the detections, each shaped "
            '{ "id": <0-based index>, "verdict": "confirmed" | "false_positive", "reason": "<short reason>", '
            '"box"?: { "x":0-1, "y":0-1, "w":0-1, "h":0-1 } }.'
        )

        content = [
            {
                "type": "text",
                "text": f"Verify these {len(entries)} detection(s). The crops follow in order.\n\n" + "\n".join(lines),
            }
        ] + [{"type": "image_url", "image_url": {"url": e["ctx"]["crop"]}} for e in entries]

        try:
            raw = await llm_mod.chat_completion(
                provider=llm["provider"],
                model=llm["model"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                temperature=0,
                max_tokens=120 * len(entries) + 200,
            )
            verdicts = parse_model_json(raw)
            for v in verdicts if isinstance(verdicts, list) else []:
                if not isinstance(v, dict):
                    continue
                try:
                    target = entries[int(v.get("id"))]
                except (TypeError, ValueError, IndexError):
                    continue
                verdict = re.sub(r"[^a-z]", "", str(v.get("verdict") or "").lower())
                is_false_positive = verdict == "falsepositive"
                target["issue"]["verified"] = not is_false_positive
                reason = v.get("reason")
                if isinstance(reason, str) and reason.strip():
                    target["issue"]["verification_note"] = reason.strip()[:200]
                # Apply a box correction only for confirmed detections, and only
                # when it substantially disagrees with the current (detector)
                # box — the detector localizes better than the VLM, so minor
                # nudges are ignored.
                if not is_false_positive and "box" in v:
                    corrected = map_crop_box_to_image(v.get("box"), target["ctx"]["region"])
                    current = target["issue"].get("bbox")
                    substantial = corrected and (not current or iou_boxes(current, corrected) < correction_max_iou)
                    if corrected and substantial:
                        target["issue"]["bbox"] = corrected
                        # The old polygon was traced for the detector box; it no
                        # longer matches the corrected box, so drop it.
                        target["issue"]["polygon"] = None
                        target["issue"]["box_adjusted"] = True
        except Exception as verify_error:
            print(f"[merchandise] detection verification failed, keeping unverified: {verify_error}")

    await asyncio.gather(*[process_result(r) for r in results])


# --------------------------------------------------------------------------- #
# The class.
# --------------------------------------------------------------------------- #


class DamageDetector:
    """Runs the merchandise damage-detection pipeline on one or more images.

    Ported line-for-line from the CURRENT logic in
    nextjs-app/app/api/merchandise/analyze/route.ts's POST handler: the same
    LLM prompts, the same detector-first CV sweep via the localizer service,
    the same box refinement/verification/dedupe — organized as a class so it
    can be instantiated once (with a fixture config) and called once per
    image, e.g. from a multi-image alignment pipeline that needs each photo's
    detections before merging overlaps across photos of the same table.
    """

    def __init__(
        self,
        fixture_config: dict,
        *,
        fixture_type_id: Optional[str] = None,
        model: Optional[str] = None,
        localizer_url: Optional[str] = None,
        localizer_timeout_ms: Optional[float] = None,
        detector_first: Optional[bool] = None,
        localizer_verify: Optional[bool] = None,
        correction_max_iou: Optional[float] = None,
    ) -> None:
        self.fixture_config = fixture_config
        self.fixture_type_id = fixture_type_id
        self.model = model
        self.localizer_base = (
            localizer_url if localizer_url is not None else os.getenv("LOCALIZER_URL", "")
        ).strip().rstrip("/")
        self.localizer_timeout_ms = (
            localizer_timeout_ms if localizer_timeout_ms is not None else float(os.getenv("LOCALIZER_TIMEOUT_MS") or 20000)
        )
        self.detector_first_enabled = (
            detector_first if detector_first is not None else os.getenv("DETECTOR_FIRST") != "0"
        )
        self.localizer_verify_enabled = (
            localizer_verify if localizer_verify is not None else os.getenv("LOCALIZER_VERIFY") != "0"
        )
        self.correction_max_iou = (
            correction_max_iou if correction_max_iou is not None else DEFAULT_CORRECTION_MAX_IOU
        )

    async def analyze(self, image: dict, *, observations: Optional[str] = None) -> dict:
        """Run damage detection on ONE image.

        `image`: {id, name, data_url, content_type}. Returns that image's
        VisionResult dict: {image_id, caption, detected_issues: [...],
        cleared_objects?: [...]}.
        """
        payload = await self.analyze_batch([image], observations=observations)
        return payload["results"][0]

    async def analyze_batch(self, images: list, *, observations: Optional[str] = None) -> dict:
        """Run damage detection on a batch of images sharing one merged
        prompt/photo-order (mirrors AnalyzeRequest in route.ts exactly).

        Returns {provider, model, baseUrl, results, localizer, detector_first}.
        """
        if not images:
            raise DamageDetectionError("missing_images", code="missing_images")
        if not (self.fixture_config.get("issue_taxonomy") or []):
            raise DamageDetectionError("missing_fixture_config", code="missing_fixture_config")

        reference_items = collect_reference_items(self.fixture_config)
        audit_urls = [img["data_url"] for img in images]
        reference_urls = [ref["data_url"] for ref in reference_items]
        primary_images = [*reference_urls, *audit_urls]
        if reference_items:
            primary_user_text = (
                f"The first {len(reference_items)} image(s) are REFERENCE examples — do NOT flag issues in "
                f"them. Analyze the remaining {len(images)} AUDIT photo(s) in order and return exactly "
                f"{len(images)} JSON entries."
            )
        else:
            primary_user_text = f"Analyze these {len(images)} fixture audit photo(s) in order."

        llm: Optional[dict] = None
        model_label = (self.model or "").strip() or "(env default)"
        try:
            llm = await llm_mod.resolve_llm("vision", self.model)
            model_label = llm["model"]

            localizer_base = self.localizer_base if self.detector_first_enabled else ""
            detector_labels = collect_detector_labels(self.fixture_config) if localizer_base else []
            use_merged = len(detector_labels) > 0

            detector_first_stats: Optional[dict] = None
            if use_merged:
                outcomes = await asyncio.gather(
                    *[
                        _analyze_image_merged(
                            llm,
                            self.fixture_config,
                            self.fixture_type_id,
                            observations,
                            image,
                            reference_items,
                            detector_labels,
                            localizer_base,
                            self.localizer_timeout_ms,
                        )
                        for image in images
                    ]
                )
                results = [o["result"] for o in outcomes]
                detector_first_stats = {r["image_id"]: o["stats"] for r, o in zip(results, outcomes)}
            else:
                system = build_system_prompt(self.fixture_config, self.fixture_type_id, observations, images, reference_items)
                content = await _call_llm_vision(llm, system, primary_images, primary_user_text)
                try:
                    results = _parse_vision_results(content, images)
                except Exception as parse_error:
                    raise DamageDetectionError(
                        "The LLM did not return valid JSON.",
                        code="model_json_parse_failed",
                        detail=content[:1200],
                        provider=llm["provider"],
                        model=llm["model"],
                        base_url=llm.get("baseUrl"),
                    ) from parse_error

            # The merged pass inspects every handset individually, so the extra
            # phone-state pass is only worth its tokens when marks are off,
            # produced no candidates, or don't cover handsets at all.
            detector_covers_phones = bool(
                detector_first_stats
                and any(re.search(r"phone|tablet|handset", label, re.I) for label in detector_labels)
                and any(s["candidates"] > 0 for s in detector_first_stats.values())
            )

            if not _has_powered_off_phone(results) and not detector_covers_phones:
                try:
                    phone_state_content = await _call_llm_vision(
                        llm,
                        build_phone_state_verification_prompt(self.fixture_config, self.fixture_type_id, observations, images),
                        audit_urls,
                        f"Inspect the {len(images)} fixture audit photo(s) in order for powered-off demo handsets.",
                    )
                    phone_state_results = _parse_vision_results(phone_state_content, images)
                    results = _merge_phone_verification_results(results, phone_state_results)
                except Exception as verification_error:
                    print(f"[merchandise] phone-state verification failed: {verification_error}")

            # Refine coarse LLM boxes via the CV localizer, and (unless
            # disabled) ask it for tight crops so a verification pass can veto
            # false positives.
            want_verify = self.localizer_verify_enabled
            localizer_outcome = await _localize_vision_results(
                results, images, want_verify, localizer_base, self.localizer_timeout_ms
            )

            # Dedupe AFTER scene boxes were refined (the check compares boxes)
            # but BEFORE verification, so dropped duplicates don't burn
            # verifier tokens: a scene finding of the same type on the same
            # object as a mark-judged finding is the same physical problem —
            # the mark's verdict wins (better box, judged on its own pixels).
            if use_merged:
                for result in results:
                    mark_issues = [i for i in result["detected_issues"] if i.get("origin") == "detector_first"]
                    if not mark_issues:
                        continue
                    result["detected_issues"] = [
                        issue
                        for issue in result["detected_issues"]
                        if issue.get("origin") == "detector_first"
                        or not any(
                            mark["issue_type_id"] == issue["issue_type_id"]
                            and issue.get("bbox")
                            and mark.get("bbox")
                            and iou_boxes(issue["bbox"], mark["bbox"]) > DETECTOR_DEDUPE_IOU
                            for mark in mark_issues
                        )
                    ]

            df_candidates = df_issues = df_cleared = df_rejected = df_bad_box = 0
            if detector_first_stats:
                for stats in detector_first_stats.values():
                    df_candidates += stats["candidates"]
                    df_issues += stats["issues"]
                    df_cleared += stats["cleared"]
                    df_rejected += stats["rejected"]
                    df_bad_box += stats["bad_box"]

            if want_verify and localizer_outcome["crops"]:
                await _verify_detections(llm, results, localizer_outcome["crops"], self.fixture_config, self.correction_max_iou)

            verified_count = 0
            flagged_count = 0
            for r in results:
                for issue in r["detected_issues"]:
                    if issue.get("verified") is True:
                        verified_count += 1
                    elif issue.get("verified") is False:
                        flagged_count += 1

            return {
                "provider": llm["provider"],
                "model": llm["model"],
                "baseUrl": llm.get("baseUrl"),
                "results": results,
                # Diagnostics so a silently-degraded run is visible: when
                # `configured && not reachable`, boxes are coarse LLM boxes —
                # start the localizer service (see localizer/README.md).
                "localizer": {
                    "configured": localizer_outcome["configured"],
                    "reachable": localizer_outcome["reachable"],
                    "refined": localizer_outcome["refined"],
                    "total": localizer_outcome["total"],
                    "verified": verified_count,
                    "flagged_false_positive": flagged_count,
                },
                "detector_first": {
                    "enabled": use_merged,
                    "mode": "merged_set_of_marks",
                    "ran": detector_first_stats is not None,
                    "labels": detector_labels,
                    "candidates": df_candidates,
                    "issues": df_issues,
                    "cleared": df_cleared,
                    "rejected_not_object": df_rejected,
                    "bad_box": df_bad_box,
                },
            }
        except DamageDetectionError:
            raise
        except Exception as error:  # noqa: BLE001 - shaped into the error contract, mirrors route.ts's catch
            status = getattr(error, "status", None)
            detail = getattr(error, "detail", None)
            detail = detail if isinstance(detail, str) else ""
            code = getattr(error, "code", None)
            provider = getattr(error, "provider", None) or (llm["provider"] if llm else "llm")
            base_url = getattr(error, "base_url", None) or (llm.get("baseUrl") if llm else None)
            provider_label = llm_mod.llm_provider_label(provider)

            if isinstance(status, int):
                raise DamageDetectionError(
                    f"{provider_label} returned {status}",
                    code=code or f"{provider}_error",
                    status=status,
                    detail=detail[:500]
                    or (
                        f'{provider_label} is reachable, but the request failed. Check that model '
                        f'"{model_label}" exists and supports vision.'
                    ),
                    provider=provider,
                    model=llm["model"] if llm else model_label,
                    base_url=base_url,
                ) from error

            raise DamageDetectionError(
                llm_mod.llm_setup_message(provider)
                if code in ("missing_api_key", "invalid_provider") or provider == "local"
                else f"Could not complete the {provider_label} request.",
                code=code or f"{provider}_error",
                detail=str(error),
                provider=provider,
                model=llm["model"] if llm else model_label,
                base_url=base_url,
            ) from error
