"""One command for a whole fixture audit: detect, merge, report.

Wraps the two halves of the pipeline that are otherwise run separately —
`run_damage_detector.py` (costs LLM tokens) and `multiview_merge.py` (free) —
and prints the numbers that matter: how many distinct devices have problems,
how many findings that is, and how many raw detections were duplicates of each
other across overlapping photos.

    python run_audit.py --images 'photo*.jpg'

The detector's payload is cached to <out>/detections.json, so re-running with
--reuse re-does only the merge. That matters because thresholds are worth
sweeping and detection is not worth paying for twice:

    python run_audit.py --images 'photo*.jpg' --reuse --min-confidence 0.6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import multiview_merge as mv
from merch_ai.damage_detector import DamageDetector
from run_damage_detector import DEFAULT_CONFIG, collect_images, fetch_products, to_data_url

RULE = "=" * 70


def _fmt_pct(x: float) -> str:
    return f"{x:.0%}"


async def detect(images: list, fixture_config: dict, args) -> tuple:
    """Run the detector over every photo; returns (payload, products)."""
    detector = DamageDetector(
        fixture_config,
        fixture_type_id=args.fixture_type,
        model=args.model,
        localizer_url=args.localizer_url or None,
    )
    payload = await detector.analyze_batch(images, observations=args.observations)

    products = None
    if args.localizer_url:
        labels = []
        for issue in fixture_config.get("issue_taxonomy") or []:
            for label in issue.get("object_labels") or []:
                if label and label not in labels:
                    labels.append(label)
        products = await fetch_products(images, labels or ["product"], args.localizer_url, 20000.0)
    return payload, products


def report(rep: dict, payload: dict, args) -> None:
    a, c = rep["alignment"], rep["counts"]
    cap = a["capture"]

    print(f"\n{RULE}\n FIXTURE AUDIT — {len(a['aligned']) + len(a['unaligned'])} photo(s)\n{RULE}")

    print(" INPUT")
    print(f"   model            : {payload['provider']}/{payload['model']}")
    loc = payload["localizer"]
    print(f"   localizer        : {'reachable' if loc['reachable'] else 'not configured'}"
          f"   (boxes {'refined by detector' if loc['refined'] else 'are coarse LLM boxes'})")

    print("\n GEOMETRY")
    print(f"   aligned          : {', '.join(a['aligned'])}")
    if a["unaligned"]:
        print(f"   UNALIGNED        : {', '.join(a['unaligned'])}  (excluded from every count below)")
    print(f"   alignment error  : {a['rms_px']} px ({(a['rms_fraction_of_canvas'] or 0):.2%} of canvas)"
          f"   {'OK' if a['trustworthy'] else '<-- HIGH, counts unreliable'}")
    print(f"   capture          : {cap['verdict']}  (weakest overlap {_fmt_pct(cap['weakest_overlap'])})")
    for line in cap["advice"]:
        print(f"                      ! {line}")
    if a["unverified_edges"]:
        pairs = ", ".join(f"{x}-{y}" for x, y in a["unverified_edges"])
        print(f"   unverified links : {pairs}  (no third photo corroborates these)")

    print("\n RESULTS")
    print(f"   devices with issues : {c['distinct_devices']}")
    print(f"   bounding boxes      : {c['distinct_issues']}")
    dropped = (f", {c['below_confidence_floor']} below confidence floor"
               if c.get("below_confidence_floor") else "")
    print(f"   raw detections      : {c['issue_observations']}"
          f"   ({c['duplicates_removed']} merged as duplicates{dropped})")
    if c["distinct_products"]:
        print(f"   products on fixture : {c['distinct_products']}"
              f"   (from {c['product_observations']} observations)")

    if rep["devices"]:
        print("\n PER DEVICE")
        for dev in rep["devices"]:
            issues = [i for i in rep["issues"] if i["device_id"] == dev["device_id"]]
            n_photos = len(dev["detected_in"])
            tag = "corroborated" if n_photos > 1 else "single view"
            if any(i["low_support"] for i in issues):
                tag = "LOW SUPPORT — review"
            print(f"   device {dev['device_id']}  conf {dev['confidence']:.2f}  "
                  f"seen in {n_photos} photo(s)  [{tag}]")
            for i in issues:
                where = ", ".join(o["image_id"] for o in i["observations"])
                print(f"       {i['issue_type_id']}  conf {i['merged_confidence']:.2f}"
                      f"   merged from {len(i['observations'])} box(es): {where}")
                b = i.get("bbox")
                if b:
                    # Normalized 0-1 in its own photo, so it can be drawn on
                    # that image directly without knowing the pixel size.
                    print(f"         box   ({i['bbox_image_id']})  "
                          f"x={b['x']:.3f} y={b['y']:.3f} w={b['w']:.3f} h={b['h']:.3f}")
                cb = i.get("canonical_bbox")
                if cb:
                    print(f"         table  x={cb['x']:.0f} y={cb['y']:.0f} "
                          f"w={cb['w']:.0f} h={cb['h']:.0f}  (canonical px)")
                for line in i.get("evidence_all") or ([i["evidence"]] if i.get("evidence") else []):
                    print(f"         \"{line}\"")

    print(f"\n wrote {os.path.abspath(args.out)}/\n{RULE}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory or glob of photos of ONE fixture")
    ap.add_argument("--out", default="audit_out", help="output directory (default: audit_out)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse <out>/detections.json instead of calling the LLM again")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop merged findings below this confidence (applied after cross-view voting)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="fixture config JSON")
    ap.add_argument("--fixture-type", default="demo_table")
    ap.add_argument("--observations", default=None, help="free-text context passed to the model")
    ap.add_argument("--model", default=None, help="override the vision model")
    ap.add_argument("--localizer-url", default=os.getenv("LOCALIZER_URL", ""))
    ap.add_argument("--extractor", default="superpoint", choices=["superpoint", "disk", "aliked", "sift"])
    ap.add_argument("--no-viz", action="store_true", help="skip the canonical mosaic render")
    args = ap.parse_args()

    paths = collect_images(args.images)
    if len(paths) < 2:
        print(f"need at least 2 images, found {len(paths)} at {args.images}", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    det_path = os.path.join(args.out, "detections.json")
    prod_path = os.path.join(args.out, "products.json")

    with open(args.config) as fh:
        fixture_config = json.load(fh)

    products = None
    if args.reuse and os.path.exists(det_path):
        print(f"[audit] reusing {det_path} (no LLM calls)")
        with open(det_path) as fh:
            payload = json.load(fh)
        if os.path.exists(prod_path):
            with open(prod_path) as fh:
                products = json.load(fh)
    else:
        images = []
        for path in paths:
            url, mime = to_data_url(path)
            images.append({"id": os.path.splitext(os.path.basename(path))[0],
                           "name": os.path.basename(path), "data_url": url, "content_type": mime})
        print(f"[audit] detecting on {len(images)} photo(s) — this calls the LLM")
        payload, products = await detect(images, fixture_config, args)
        with open(det_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        if products:
            with open(prod_path, "w") as fh:
                json.dump(products, fh, indent=2)

    cfg = mv.MergeConfig(extractor=args.extractor, min_confidence=args.min_confidence,
                         localizer_url=args.localizer_url)
    rep = mv.run(paths, payload, products_payload=products, cfg=cfg, verbose=False)

    with open(os.path.join(args.out, "merged.json"), "w") as fh:
        json.dump(mv._json_safe(rep), fh, indent=2)
    if not args.no_viz:
        import cv2

        g = rep["_geometry"]
        cv2.imwrite(os.path.join(args.out, "canonical.png"),
                    mv.render_canvas(g["views"], g["transforms"], g["polys"], rep, g["size"]))

    report(rep, payload, args)
    # Non-zero when the geometry can't be trusted, so a caller can gate on it.
    return 0 if rep["alignment"]["trustworthy"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
