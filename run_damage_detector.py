"""Run the damage detector over a set of photos and write the analyze_batch
payload that `multiview_merge.py` consumes.

The two halves of the audit pipeline are deliberately separate processes: this
one costs LLM tokens and needs network, the merge does not. Keeping the payload
on disk between them means the merge can be re-run and re-thresholded as often
as you like without paying for detection again.

    python run_damage_detector.py --images photos/ --out detections.json
    python multiview_merge.py --images photos/ --detections detections.json --viz

Products for the merge's inventory counting come from the same localizer the
detector-first sweep uses; pass --products-out to capture them in one pass while
the images are already encoded.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import mimetypes
import os
import sys

from merch_ai.damage_detector import DamageDetector

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merch_ai", "fixture_config.json")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def to_data_url(path: str) -> tuple:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return "data:" + mime + ";base64," + base64.b64encode(fh.read()).decode(), mime


def collect_images(spec: str) -> list:
    paths = (
        sorted(p for p in glob.glob(os.path.join(spec, "*")) if p.lower().endswith(IMAGE_EXTS))
        if os.path.isdir(spec)
        else sorted(glob.glob(spec))
    )
    return paths


async def fetch_products(images: list, labels: list, base: str, timeout_ms: float) -> dict:
    """Localizer `/detect` candidates per image — the merge's product inventory.

    Reuses the already-encoded data URLs rather than re-reading the files, since
    the detector run has just paid that cost.
    """
    import httpx

    out: dict = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_ms / 1000.0)) as client:
        for image in images:
            try:
                res = await client.post(
                    base.rstrip("/") + "/detect",
                    json={"image_data_url": image["data_url"], "labels": labels},
                )
                if res.status_code >= 400:
                    print(f"[detector] localizer {res.status_code} for {image['id']}")
                    continue
                cands = res.json().get("candidates") or []
            except Exception as error:  # noqa: BLE001
                print(f"[detector] localizer unreachable for {image['id']}: {error}")
                continue
            out[image["id"]] = [
                {"label": str(c.get("label") or "object"), "score": float(c.get("score") or 0.0), "box": c["box"]}
                for c in cands
                if isinstance(c.get("box"), dict)
            ]
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory or glob of photos of ONE fixture")
    ap.add_argument("--out", required=True, help="where to write the analyze_batch payload")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="fixture config JSON")
    ap.add_argument("--fixture-type", default="demo_table")
    ap.add_argument("--observations", default=None, help="free-text context passed to the model")
    ap.add_argument("--model", default=None, help="override the vision model")
    ap.add_argument("--localizer-url", default=os.getenv("LOCALIZER_URL", ""))
    ap.add_argument("--products-out", help="also write localizer /detect candidates here")
    args = ap.parse_args()

    paths = collect_images(args.images)
    if not paths:
        print(f"no images found at {args.images}", file=sys.stderr)
        return 2

    with open(args.config) as fh:
        fixture_config = json.load(fh)

    images = []
    for path in paths:
        url, mime = to_data_url(path)
        images.append(
            {
                "id": os.path.splitext(os.path.basename(path))[0],
                "name": os.path.basename(path),
                "data_url": url,
                "content_type": mime,
            }
        )
    print(f"[detector] {len(images)} image(s): {', '.join(i['id'] for i in images)}")

    detector = DamageDetector(
        fixture_config,
        fixture_type_id=args.fixture_type,
        model=args.model,
        localizer_url=args.localizer_url or None,
    )
    payload = await detector.analyze_batch(images, observations=args.observations)

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    loc = payload["localizer"]
    print(f"[detector] provider={payload['provider']} model={payload['model']}")
    print(
        f"[detector] localizer configured={loc['configured']} reachable={loc['reachable']} "
        f"refined={loc['refined']}/{loc['total']}"
    )
    if not loc["configured"]:
        # Coarse LLM boxes degrade both box quality and the merge's ability to
        # count products, so say it rather than letting it pass unnoticed.
        print("[detector] NOTE: no localizer — boxes are coarse LLM boxes and product counting is unavailable")
    for result in payload["results"]:
        print(f"\n  {result['image_id']}: {len(result['detected_issues'])} issue(s)")
        for issue in result["detected_issues"]:
            box = issue.get("bbox")
            loc_s = f"[{box['x']:.2f},{box['y']:.2f},{box['w']:.2f},{box['h']:.2f}]" if box else "no-box"
            print(f"    - {issue['issue_type_id']:<28} conf={issue['confidence']:.2f} {loc_s}")
    print(f"\n[detector] wrote {args.out}")

    if args.products_out and args.localizer_url:
        labels = []
        for issue in fixture_config.get("issue_taxonomy") or []:
            for label in issue.get("object_labels") or []:
                if label and label not in labels:
                    labels.append(label)
        products = await fetch_products(images, labels or ["product"], args.localizer_url, 20000.0)
        with open(args.products_out, "w") as fh:
            json.dump(products, fh, indent=2)
        print(f"[detector] wrote {args.products_out} ({sum(len(v) for v in products.values())} candidates)")
    elif args.products_out:
        print("[detector] --products-out needs --localizer-url; skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
