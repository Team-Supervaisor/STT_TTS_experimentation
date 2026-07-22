"""Multi-view merge for the merchandise audit pipeline.

Takes N overlapping handheld photos of one fixture (a showroom table) plus the
per-photo output of `damage_detector.DamageDetector.analyze_batch`, and answers
the two questions a per-photo detector cannot:

  1. How many DISTINCT products are on the table? (the same phone shot from
     three angles must count once, not three times)
  2. Which damage findings are the same physical problem seen twice, and which
     survive cross-view scrutiny?

Approach — everything is anchored to the plane of the table:

    SuperPoint + LightGlue (SIFT/BF fallback)
                  |
    pairwise homography, MAGSAC + geometric sanity checks
                  |
    alias rejection: loop closure + photometric tie-break
                  |
    overlap graph -> hub or spanning tree -> global least-squares refinement
                  |
    every view gets H_i: its pixels -> one canonical table frame
                  |
    products/damages projected by GROUND-CONTACT point, not box centre
                  |
    constrained single-linkage clustering (no two obs from the same photo)
                  |
    visibility-aware voting -> merged confidence

Two design points worth knowing before touching this file:

* A homography is exact only on a plane. The table IS that plane, so an
  object's parallax error grows with its height above it. Every projection
  here therefore uses the bottom-edge midpoint of a box — the point physically
  touching the table — which maps exactly from any viewpoint. Using the box
  centre instead reintroduces height-dependent error and breaks clustering for
  tall boxed stock.

* Showroom tables are rows of near-identical products, which is the classic
  alias failure for feature matching: a confident homography that lines up the
  wrong phone with the wrong phone. Inlier counts do NOT catch this — in
  `multiview_merge_selftest.py`'s HARD scenario the aliased edge is 365 px
  wrong while carrying a HIGHER inlier ratio (0.66) than a correct edge (0.48).
  What catches it is `prune_inconsistent_cycles`: a loop that fails to close
  proves some edge is wrong, and `photometric_score` says which. Disable either
  and the merge fails silently rather than loudly — watch `alignment.rms_px`.

Detections are consumed, not produced: this module never calls an LLM. It
reads `damage_detector`'s JSON so it can run offline, be re-run with different
thresholds, and be tested without a model. Product candidates come from the
same localizer `/detect` the detector-first sweep already uses.

CLI:
    # pre-flight: grade the capture before spending any VLM tokens.
    # exits non-zero when the photos are too sparse to verify themselves.
    python multiview_merge.py --images photos/ --check

    python multiview_merge.py --images photos/ --detections analyze.json \
        --out merged/ --viz
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from shapely.affinity import scale as _affine_scale
from shapely.geometry import Polygon
from shapely.ops import unary_union

# --------------------------------------------------------------------------- #
# Configuration.
# --------------------------------------------------------------------------- #


@dataclass
class MergeConfig:
    """Every threshold in the pipeline, in one place.

    Defaults are tuned for ~12 MP phone photos of a 2-3 m fixture shot from
    standing height with 40-60% overlap between consecutive frames.
    """

    # -- feature extraction / matching --
    extractor: str = "superpoint"       # superpoint | disk | aliked | sift
    max_keypoints: int = 2048
    resize_long_side: int = 1024        # extraction resolution; kps rescale to full res
    device: str = "auto"                # auto | cuda | cpu
    min_matches: int = 25               # below this a pair isn't worth a homography

    # -- pairwise homography --
    ransac_thresh_px: float = 4.0       # at full resolution
    min_inliers: int = 40               # absolute inlier count to admit an edge
    min_inlier_ratio: float = 0.30      # guards against the identical-product alias
    max_area_ratio: float = 6.0         # warped frame vs. own frame; catches horizon blowups
    max_shear: float = 0.55             # |cos(angle)| between warped frame axes

    # -- alias rejection --
    # Inlier count does NOT discriminate a one-slot alias on identical stock:
    # measured on a synthetic showroom, the WRONG edge scored 0.66 inlier ratio
    # against 0.48 for a correct one. These two checks do the real work.
    photometric_check: bool = True
    photometric_scale: float = 0.25     # ZNCC is computed downsampled; it's a coarse check
    min_photometric: float = 0.35       # permissive floor: only catches gross mismatches
    cycle_check: bool = True
    cycle_tolerance_frac: float = 0.02  # loop-closure error, as a fraction of image diagonal

    # -- capture quality --
    # Above 50% consecutive overlap, photo i and photo i+2 share ground, so the
    # closing edge exists and the graph can verify itself. Below it, a chain
    # capture has no redundancy at all. 0.55 warns just past the cliff; 0.65 is
    # what the operator is told to aim for.
    min_consecutive_overlap: float = 0.55
    target_consecutive_overlap: float = 0.65

    # -- global alignment --
    refine_global: bool = True
    refine_max_pts_per_edge: int = 250  # subsample correspondences for speed
    canvas_max_side: int = 2000         # canonical frame is rescaled to fit this

    # -- clustering --
    # Radii are fractions of the MEDIAN PRODUCT WIDTH in canonical units, so
    # they survive the arbitrary scale of the table plane frame.
    product_cluster_frac: float = 0.55
    # Fallback only (no product candidates): findings merge on canonical box
    # overlap rather than a guessed radius. Measured on real showroom photos,
    # same-phone pairs scored 0.32 and unrelated pairs 0.00, so anything in
    # between is safe; 0.15 leans toward merging when boxes are coarse.
    # Along-fixture overlap is the PRIMARY identity signal. Devices on a
    # fixture are separated horizontally, and that is also the axis a VLM gets
    # right: measured on real showroom photos, two views of one phone scored
    # X-IoU 0.91 while every unrelated pair scored 0.00 — a margin that
    # 2-D IoU (0.34 vs 0.00) does not come close to, because almost all of the
    # box error is vertical.
    damage_x_iou_threshold: float = 0.35
    # Weak vertical gate, not the main signal. Its only job is to stop two
    # devices in DIFFERENT rows of a multi-row fixture — same X, very different
    # depth — from merging on horizontal overlap alone.
    damage_iou_threshold: float = 0.10
    # Rolling findings up to devices: looser than the dedupe threshold, since
    # two different issue types on one handset (cracked screen + dark display)
    # get boxed differently by the detector and overlap less than two views of
    # the same finding do.
    device_iou_threshold: float = 0.10
    # Vertical slack applied before any box-overlap test, as a fraction of box
    # height. A VLM's box for a standing object is far more reliable in X (where
    # neighbouring devices are actually separated) than in Y: on real showroom
    # photos two views of ONE phone agreed to 6 px horizontally but disagreed by
    # ~115 px vertically, which drove 2-D IoU to zero and split the device in
    # two. Dilating vertically absorbs that error without merging rows of a
    # multi-row fixture, which sit much further apart than one object height.
    box_vertical_slack: float = 1.0
    require_label_match: bool = True    # two obs only merge if their labels agree

    # -- visibility / voting --
    visibility_margin_px: float = 40.0  # erode footprints; frame-edge views don't vote
    vote_floor: float = 0.55            # merged conf floor multiplier at zero agreement
    low_support_ratio: float = 0.50     # at/below this agreement, flag the finding for review

    # -- io --
    # Reporting floor. An audit surfaces findings a human will act on, so
    # near-threshold guesses are noise rather than coverage. Applied AFTER
    # cross-view voting, so a finding corroborated by several photos survives a
    # cut that a lone low-confidence guess does not.
    min_confidence: float = 0.0
    localizer_url: str = ""
    localizer_timeout_ms: float = 20000.0

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


# --------------------------------------------------------------------------- #
# Views — one loaded photo plus its identity in the detection JSON.
# --------------------------------------------------------------------------- #


@dataclass
class View:
    index: int
    image_id: str
    path: str
    bgr: np.ndarray = field(repr=False)

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])

    def frame_quad(self) -> np.ndarray:
        """The image's own corners, clockwise from top-left, as (4,1,2)."""
        w, h = float(self.width), float(self.height)
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def load_views(paths: list, ids: Optional[list] = None) -> list:
    """Load images in the given order. cv2.imread applies EXIF orientation, so
    pixel coordinates here match what the VLM was shown — that equivalence is
    what lets normalized detection boxes be scaled by (width, height)."""
    views = []
    for i, path in enumerate(paths):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read image: {path}")
        image_id = ids[i] if ids and i < len(ids) else os.path.splitext(os.path.basename(path))[0]
        views.append(View(index=i, image_id=str(image_id), path=path, bgr=bgr))
    return views


# --------------------------------------------------------------------------- #
# Feature extraction + matching. SuperPoint/LightGlue when available, SIFT/BF
# otherwise, behind one interface so the rest of the file doesn't care.
# --------------------------------------------------------------------------- #


class FeatureMatcher:
    """Extracts once per image, matches on demand. Keypoints are always
    returned in FULL-RESOLUTION pixel coordinates regardless of the resolution
    features were computed at."""

    def __init__(self, cfg: MergeConfig) -> None:
        self.cfg = cfg
        self.backend = "sift"
        self._cache: dict = {}
        if cfg.extractor != "sift":
            try:
                self._init_lightglue()
                self.backend = "lightglue"
            except Exception as error:  # noqa: BLE001 - degrading is the point
                print(f"[multiview] LightGlue unavailable ({error}); falling back to SIFT")
        if self.backend == "sift":
            self._sift = cv2.SIFT_create(nfeatures=cfg.max_keypoints)
            self._bf = cv2.BFMatcher(cv2.NORM_L2)

    def _init_lightglue(self) -> None:
        import torch
        from lightglue import ALIKED, DISK, SuperPoint, LightGlue

        self._torch = torch
        self.device = self.cfg.resolved_device()
        kinds = {"superpoint": SuperPoint, "disk": DISK, "aliked": ALIKED}
        kind = kinds.get(self.cfg.extractor)
        if kind is None:
            raise ValueError(f"unknown extractor {self.cfg.extractor!r}")
        self._extractor = kind(max_num_keypoints=self.cfg.max_keypoints).eval().to(self.device)
        self._matcher = LightGlue(features=self.cfg.extractor).eval().to(self.device)

    # -- extraction ------------------------------------------------------- #

    def features(self, view: View) -> Any:
        if view.index not in self._cache:
            self._cache[view.index] = (
                self._extract_lightglue(view) if self.backend == "lightglue" else self._extract_sift(view)
            )
        return self._cache[view.index]

    def _extract_lightglue(self, view: View):
        rgb = cv2.cvtColor(view.bgr, cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self.device)
        # Extractor.extract resizes internally and rescales keypoints back to
        # the coordinates of the tensor it was handed — i.e. full resolution.
        return self._extractor.extract(tensor, resize=self.cfg.resize_long_side)

    def _extract_sift(self, view: View):
        gray = cv2.cvtColor(view.bgr, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.cfg.resize_long_side / max(gray.shape))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
        kps, desc = self._sift.detectAndCompute(small, None)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2) / scale
        return {"keypoints": pts, "descriptors": desc}

    def keypoint_count(self, view: View) -> int:
        feats = self.features(view)
        if self.backend == "lightglue":
            return int(feats["keypoints"].shape[1])
        return int(len(feats["keypoints"]))

    # -- matching --------------------------------------------------------- #

    def match(self, a: View, b: View) -> tuple:
        """Return (pts_a, pts_b) as (K,2) float32 arrays of corresponding
        full-resolution points."""
        if self.backend == "lightglue":
            return self._match_lightglue(a, b)
        return self._match_sift(a, b)

    def _match_lightglue(self, a: View, b: View) -> tuple:
        from lightglue.utils import rbd

        fa, fb = self.features(a), self.features(b)
        with self._torch.no_grad():
            out = self._matcher({"image0": fa, "image1": fb})
        m = rbd(out)["matches"]  # (K, 2) index pairs
        ka, kb = rbd(dict(fa))["keypoints"], rbd(dict(fb))["keypoints"]
        pts_a = ka[m[:, 0]].cpu().numpy().astype(np.float32)
        pts_b = kb[m[:, 1]].cpu().numpy().astype(np.float32)
        return pts_a, pts_b

    def _match_sift(self, a: View, b: View) -> tuple:
        fa, fb = self.features(a), self.features(b)
        if fa["descriptors"] is None or fb["descriptors"] is None:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        raw = self._bf.knnMatch(fa["descriptors"], fb["descriptors"], k=2)
        # Lowe's ratio test at 0.75 — deliberately strict, because a showroom
        # of identical products produces many near-tied second-best matches.
        good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
        pts_a = np.array([fa["keypoints"][m.queryIdx] for m in good], np.float32).reshape(-1, 2)
        pts_b = np.array([fb["keypoints"][m.trainIdx] for m in good], np.float32).reshape(-1, 2)
        return pts_a, pts_b


# --------------------------------------------------------------------------- #
# Pairwise homography.
# --------------------------------------------------------------------------- #


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N,2) points, returning (N,2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), np.float64)
    return cv2.perspectiveTransform(pts, H.astype(np.float64)).reshape(-1, 2)


def sanity_homography(H: np.ndarray, src: View, dst: View, cfg: MergeConfig) -> Optional[str]:
    """Reject homographies that are numerically fine but physically absurd.

    This is the guard against the identical-product alias: RANSAC will happily
    return a high-inlier H that maps one row of phones onto the next row, and
    the giveaway is almost always a degenerate warped frame — folded, mirrored,
    wildly rescaled, or stretched toward the horizon. Returns a reason string
    when the homography should be dropped, or None when it looks real.
    """
    if H is None or not np.all(np.isfinite(H)):
        return "non_finite"

    quad = project(H, src.frame_quad().reshape(-1, 2))
    if not np.all(np.isfinite(quad)):
        return "non_finite_corners"

    # Signed area via the shoelace formula: negative means the quad was
    # mirrored, near-zero means it collapsed to a line.
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area <= 0:
        return "mirrored_or_degenerate"

    own_area = float(src.width * src.height)
    ratio = area / own_area
    if ratio > cfg.max_area_ratio or ratio < 1.0 / cfg.max_area_ratio:
        return f"area_ratio={ratio:.2f}"

    # Convexity: any reflex corner means the frame folded over itself, which no
    # real camera motion over a plane can produce.
    cross_signs = []
    for i in range(4):
        p0, p1, p2 = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        u, v = p1 - p0, p2 - p1
        cross_signs.append(np.sign(u[0] * v[1] - u[1] * v[0]))
    if len(set(s for s in cross_signs if s != 0)) > 1:
        return "non_convex"

    # Shear of the local affine part; an extreme value means we're projecting
    # near the horizon where positional error explodes.
    a = H[:2, :2] / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1e-12)
    c0, c1 = a[:, 0], a[:, 1]
    n0, n1 = np.linalg.norm(c0), np.linalg.norm(c1)
    if n0 < 1e-9 or n1 < 1e-9:
        return "degenerate_affine"
    if abs(float(np.dot(c0, c1)) / (n0 * n1)) > cfg.max_shear:
        return "excessive_shear"
    return None


def photometric_score(src: View, dst: View, H: np.ndarray, scale: float) -> float:
    """Zero-mean normalized cross-correlation between `src` warped into `dst`
    and `dst` itself, over their overlap.

    This is the check that survives identical stock. A homography aliased by
    one product slot lines the products up perfectly — that is what fooled the
    matcher — but it necessarily misaligns everything that ISN'T repeated: the
    table surface, its edges, price tags, whatever is behind. Appearance sees
    that; inlier counts do not.

    Computed downsampled, and deliberately coarse: it is used to rank suspect
    edges against each other, not as a precise similarity measure.
    """
    S = np.diag([scale, scale, 1.0])
    Hs = S @ H @ np.linalg.inv(S)
    ga = cv2.cvtColor(cv2.resize(src.bgr, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(dst.bgr, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    size = (gb.shape[1], gb.shape[0])
    try:
        warped = cv2.warpPerspective(ga, Hs, size)
        mask = cv2.warpPerspective(np.ones_like(ga), Hs, size) > 0
    except cv2.error:
        return -1.0
    if int(mask.sum()) < 500:
        return -1.0  # too little overlap to judge
    x = warped[mask].astype(np.float64)
    y = gb[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0 else -1.0


@dataclass
class Pair:
    """One edge of the overlap graph: the homography taking `src` pixels into
    `dst` pixels, plus the inlier correspondences that justified it."""

    src: int
    dst: int
    H: Optional[np.ndarray]
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    pts_src: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2), np.float32))
    pts_dst: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2), np.float32))
    photo_score: float = float("nan")
    accepted: bool = False
    reason: str = ""

    def to_json(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "matches": self.n_matches,
            "inliers": self.n_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "photometric": None if self.photo_score != self.photo_score else round(self.photo_score, 4),
            "accepted": self.accepted,
            "reason": self.reason,
        }


def estimate_pair(a: View, b: View, matcher: FeatureMatcher, cfg: MergeConfig) -> Pair:
    """Estimate the homography mapping `a`'s pixels into `b`'s pixels."""
    pts_a, pts_b = matcher.match(a, b)
    n = int(len(pts_a))
    if n < cfg.min_matches:
        return Pair(a.index, b.index, None, n, 0, 0.0, reason="too_few_matches")

    H, mask = cv2.findHomography(
        pts_a.reshape(-1, 1, 2), pts_b.reshape(-1, 1, 2), cv2.USAC_MAGSAC, cfg.ransac_thresh_px
    )
    if H is None or mask is None:
        return Pair(a.index, b.index, None, n, 0, 0.0, reason="ransac_failed")

    keep = mask.ravel().astype(bool)
    n_in = int(keep.sum())
    ratio = n_in / float(n)
    pair = Pair(a.index, b.index, H, n, n_in, ratio, pts_a[keep], pts_b[keep])

    if n_in < cfg.min_inliers:
        pair.reason = f"inliers={n_in}<{cfg.min_inliers}"
        return pair
    if ratio < cfg.min_inlier_ratio:
        pair.reason = f"inlier_ratio={ratio:.2f}<{cfg.min_inlier_ratio}"
        return pair
    bad = sanity_homography(H, a, b, cfg)
    if bad:
        pair.reason = bad
        return pair

    if cfg.photometric_check:
        pair.photo_score = photometric_score(a, b, H, cfg.photometric_scale)
        if pair.photo_score < cfg.min_photometric:
            pair.reason = f"photometric={pair.photo_score:.2f}<{cfg.min_photometric}"
            return pair

    pair.accepted = True
    pair.reason = "ok"
    return pair


def build_graph(views: list, matcher: FeatureMatcher, cfg: MergeConfig, verbose: bool = True) -> list:
    """All-pairs matching. N is small (a handful of photos of one fixture), so
    the O(N^2) sweep costs far less than the risk of missing an edge that a
    sequential-only assumption would skip."""
    pairs = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            pair = estimate_pair(views[i], views[j], matcher, cfg)
            pairs.append(pair)
            if verbose:
                flag = "ok " if pair.accepted else "REJ"
                print(
                    f"[multiview] {flag} {views[i].image_id} -> {views[j].image_id}: "
                    f"{pair.n_inliers}/{pair.n_matches} inliers ({pair.inlier_ratio:.2f}) {pair.reason}"
                )
    return pairs


def triangles(live: dict):
    """Every (i, j, k) whose three edges are all present. Edges are stored with
    src < dst, so a triangle is exactly (i,j) + (j,k) + (i,k)."""
    for (i, j) in live:
        for (j2, k) in live:
            if j2 == j and (i, k) in live:
                yield i, j, k


def unverified_edges(pairs: list) -> list:
    """Accepted edges that belong to no triangle — nothing corroborates them.

    A chain capture (walk down the table, shoot as you go) produces exactly
    this: A-B, B-C, C-D and no closing edges. Each homography is then accepted
    on its own evidence alone, and an alias in it is undetectable by geometry —
    not merely unchecked. Callers get told which edges those are rather than
    being handed a report that looks equally confident everywhere.
    """
    live = {(p.src, p.dst): p for p in pairs if p.accepted}
    covered = set()
    for i, j, k in triangles(live):
        covered |= {(i, j), (j, k), (i, k)}
    return [key for key in live if key not in covered]


def prune_inconsistent_cycles(pairs: list, views: list, cfg: MergeConfig, verbose: bool = True) -> list:
    """Drop edges that a closed loop proves cannot all be right.

    An aliased homography — one that matched the wrong row of identical
    products — is individually convincing: correct-looking geometry, plenty of
    inliers. What it cannot do is agree with the rest of the graph. If going
    i->j->k lands somewhere different from going i->k directly, at least one of
    those three edges is wrong, however good each looks alone.

    Cycles say an edge is wrong; they don't say which. The photometric score
    breaks the tie, because the aliased edge is the one whose overlap doesn't
    actually look alike. Measured on the synthetic showroom: loop error 306 px
    with the culprit at ZNCC 0.74 against 0.91 for both correct edges.

    Needs a triangle to work at all — with only two photos there is no loop to
    close and an alias cannot be detected this way.
    """
    diag = float(np.mean([math.hypot(v.width, v.height) for v in views]))
    tol = cfg.cycle_tolerance_frac * diag
    grid = None

    while True:
        live = {(p.src, p.dst): p for p in pairs if p.accepted}
        bad_edges: dict = {}
        worst = 0.0
        for i, j, k in triangles(live):
            pij, pjk, pik = live[(i, j)], live[(j, k)], live[(i, k)]
            if grid is None:
                w, h = views[i].width, views[i].height
                grid = np.array(
                    [[x, y] for x in np.linspace(0, w, 5) for y in np.linspace(0, h, 5)], dtype=np.float64
                )
            # Going i->j->k must land where going i->k directly lands.
            err = float(np.median(np.linalg.norm(project(pjk.H @ pij.H, grid) - project(pik.H, grid), axis=1)))
            if err > tol:
                worst = max(worst, err)
                for key in ((i, j), (j, k), (i, k)):
                    bad_edges[key] = max(bad_edges.get(key, 0.0), err)
        if not bad_edges:
            return pairs

        # Among every edge implicated in a broken loop, the least
        # photometrically plausible one is the culprit.
        victim_key = min(bad_edges, key=lambda k: (live[k].photo_score, live[k].inlier_ratio))
        victim = live[victim_key]
        victim.accepted = False
        victim.reason = f"cycle_inconsistent(loop={bad_edges[victim_key]:.0f}px,zncc={victim.photo_score:.2f})"
        if verbose:
            print(
                f"[multiview] dropped {views[victim.src].image_id} -> {views[victim.dst].image_id}: "
                f"breaks loop closure by {bad_edges[victim_key]:.0f}px (worst loop {worst:.0f}px), "
                f"lowest ZNCC {victim.photo_score:.2f} of the implicated edges"
            )


# --------------------------------------------------------------------------- #
# Global alignment: every view -> one canonical table frame.
# --------------------------------------------------------------------------- #


def _adjacency(pairs: list, n: int) -> dict:
    adj: dict = {i: [] for i in range(n)}
    for p in pairs:
        if not p.accepted or p.H is None:
            continue
        adj[p.src].append((p.dst, p.H, p.n_inliers))
        adj[p.dst].append((p.src, np.linalg.inv(p.H), p.n_inliers))
    return adj


def choose_reference(pairs: list, n: int) -> int:
    """Pick the hub — the view sharing the most verified structure with the
    rest. If the operator happened to take one wide establishing shot, that
    shot wins naturally (it overlaps everything), and every other view then
    reaches canonical in a single hop with no chained drift. If they didn't,
    this still picks the best-connected frame."""
    score = {i: 0 for i in range(n)}
    degree = {i: 0 for i in range(n)}
    for p in pairs:
        if not p.accepted:
            continue
        for k in (p.src, p.dst):
            score[k] += p.n_inliers
            degree[k] += 1
    return max(range(n), key=lambda i: (degree[i], score[i]))


def spanning_transforms(pairs: list, n: int, ref: int) -> dict:
    """BFS out from the reference, taking the highest-inlier edge first so the
    chain that reaches each view is the most trustworthy one available.
    Returns {view_index: H mapping that view's pixels -> reference pixels}."""
    adj = _adjacency(pairs, n)
    transforms = {ref: np.eye(3)}
    # Best-first rather than plain BFS: a strong 2-hop path beats a marginal
    # 1-hop one, and drift compounds along whichever path we commit to.
    frontier = [(ref, 0)]
    visited = {ref}
    while frontier:
        frontier.sort(key=lambda t: -t[1])
        node, _ = frontier.pop(0)
        for nbr, H_node_to_nbr, inliers in sorted(adj[node], key=lambda t: -t[2]):
            if nbr in visited:
                continue
            # H_node->nbr maps node pixels into nbr pixels; we want nbr -> ref.
            transforms[nbr] = transforms[node] @ np.linalg.inv(H_node_to_nbr)
            visited.add(nbr)
            frontier.append((nbr, inliers))
    return transforms


def _params_to_transforms(params: np.ndarray, order: list, ref: int) -> dict:
    out = {ref: np.eye(3)}
    k = 0
    for idx in order:
        if idx == ref:
            continue
        h = np.append(params[k:k + 8], 1.0).reshape(3, 3)
        out[idx] = h
        k += 8
    return out


def refine_global(transforms: dict, pairs: list, views: list, ref: int, cfg: MergeConfig) -> dict:
    """Jointly re-fit all transforms by minimizing reprojection across every
    verified correspondence at once.

    The spanning tree gives each view a transform via one chain of hops, and
    every hop compounds its predecessor's error. This spreads that error over
    all edges instead, which matters most for the views furthest from the hub —
    exactly the ones whose drift would otherwise split one product into two
    clusters.
    """
    try:
        from scipy.optimize import least_squares
    except Exception as error:  # noqa: BLE001
        print(f"[multiview] scipy unavailable ({error}); keeping spanning-tree transforms")
        return transforms

    order = sorted(transforms.keys())
    if len(order) < 3:
        return transforms  # nothing to distribute

    # Work in units of the reference's long side. Raw pixel homographies are
    # badly conditioned for a least-squares solver (entries spanning 1e-6 to
    # 1e3); normalizing puts every parameter near unit scale.
    s = float(max(views[ref].width, views[ref].height))
    S = np.diag([1.0 / s, 1.0 / s, 1.0])
    S_inv = np.linalg.inv(S)

    def normalize(H):
        Hn = S @ H @ S_inv
        return Hn / (Hn[2, 2] if abs(Hn[2, 2]) > 1e-12 else 1e-12)

    seeds = {i: normalize(H) for i, H in transforms.items()}
    x0 = np.concatenate([seeds[i].ravel()[:8] for i in order if i != ref])

    edges = []
    rng = np.random.default_rng(0)  # deterministic subsampling
    for p in pairs:
        if not p.accepted or p.src not in seeds or p.dst not in seeds:
            continue
        k = len(p.pts_src)
        if k == 0:
            continue
        if k > cfg.refine_max_pts_per_edge:
            sel = rng.choice(k, cfg.refine_max_pts_per_edge, replace=False)
            edges.append((p.src, p.dst, p.pts_src[sel] / s, p.pts_dst[sel] / s))
        else:
            edges.append((p.src, p.dst, p.pts_src / s, p.pts_dst / s))
    if not edges:
        return transforms

    def residuals(params):
        Hs = _params_to_transforms(params, order, ref)
        chunks = []
        for i, j, pi, pj in edges:
            ci = project(Hs[i], pi)
            cj = project(Hs[j], pj)
            chunks.append((ci - cj).ravel())
        r = np.concatenate(chunks)
        # A blown-up parameter set can push points to infinity; keep the
        # solver in finite arithmetic rather than letting it abort.
        return np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=1e3)

    try:
        # soft_l1 so a handful of surviving mismatches can't drag the fit.
        sol = least_squares(residuals, x0, loss="soft_l1", f_scale=0.01, max_nfev=200)
    except Exception as error:  # noqa: BLE001
        print(f"[multiview] global refinement failed ({error}); keeping spanning-tree transforms")
        return transforms

    before = float(np.sqrt(np.mean(residuals(x0) ** 2)) * s)
    after = float(np.sqrt(np.mean(residuals(sol.x) ** 2)) * s)
    if not np.isfinite(after) or after > before:
        print(f"[multiview] refinement did not improve ({before:.2f} -> {after:.2f} px); keeping seeds")
        return transforms
    print(f"[multiview] global refinement: RMS reprojection {before:.2f} -> {after:.2f} px")

    refined = _params_to_transforms(sol.x, order, ref)
    return {i: S_inv @ H @ S for i, H in refined.items()}


def alignment_rms(transforms: dict, pairs: list) -> float:
    """RMS canonical-space disagreement over every verified correspondence.

    The single number that says whether to trust the merge at all. A few pixels
    means the table reconstructed cleanly. Tens of pixels means some edge is
    still wrong, and the product count downstream is fiction — so `run` warns
    rather than returning a confident-looking report built on bad geometry.
    """
    errs = []
    for p in pairs:
        if not p.accepted or p.src not in transforms or p.dst not in transforms or len(p.pts_src) == 0:
            continue
        a = project(transforms[p.src], p.pts_src)
        b = project(transforms[p.dst], p.pts_dst)
        errs.append(np.linalg.norm(a - b, axis=1))
    if not errs:
        return float("nan")
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))


def normalize_canvas(transforms: dict, views: list, cfg: MergeConfig) -> tuple:
    """Rebase the canonical frame so it starts at the origin and fits in a
    fixed-size canvas. After this, canonical coordinates ARE canvas pixels,
    which makes both the clustering radii and the debug render trivial."""
    corners = []
    for idx, H in transforms.items():
        corners.append(project(H, views[idx].frame_quad().reshape(-1, 2)))
    allc = np.concatenate(corners, axis=0)
    x0, y0 = allc.min(axis=0)
    x1, y1 = allc.max(axis=0)
    span = max(x1 - x0, y1 - y0, 1.0)
    scale = min(1.0, cfg.canvas_max_side / span)
    T = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]], dtype=np.float64)
    out = {i: T @ H for i, H in transforms.items()}
    size = (int(math.ceil((x1 - x0) * scale)) + 1, int(math.ceil((y1 - y0) * scale)) + 1)
    return out, size


# --------------------------------------------------------------------------- #
# Footprints and visibility — "which photos could have seen this spot?"
# --------------------------------------------------------------------------- #


def footprints(transforms: dict, views: list) -> dict:
    """Each view's field of view as a polygon in the canonical frame. The
    intersection of two of these is the overlap region between those photos."""
    out = {}
    for idx, H in transforms.items():
        quad = project(H, views[idx].frame_quad().reshape(-1, 2))
        poly = Polygon(quad).buffer(0)  # buffer(0) repairs any self-touching ring
        if not poly.is_empty:
            out[idx] = poly
    return out


def overlap_regions(polys: dict) -> dict:
    """{(i, j): shared polygon} for every pair of views that actually overlap.
    This is the region a caller must not double-count products in."""
    keys = sorted(polys)
    out = {}
    for a_i in range(len(keys)):
        for b_i in range(a_i + 1, len(keys)):
            a, b = keys[a_i], keys[b_i]
            inter = polys[a].intersection(polys[b])
            if not inter.is_empty and inter.area > 0:
                out[(a, b)] = inter
    return out


def erode_footprints(polys: dict, margin: float) -> dict:
    """Shrink each footprint by `margin`. The erosion is deliberate: a
    detection at the very edge of a frame is clipped, foreshortened, and often
    half out of focus, so that view gets no vote either way rather than a bad
    one. A footprint that vanishes entirely keeps its original extent."""
    out = {}
    for idx, poly in polys.items():
        shrunk = poly.buffer(-margin)
        out[idx] = poly if shrunk.is_empty else shrunk
    return out


def visible_in(probe, eroded: dict) -> list:
    """Which views fully contain `probe` — an object's canonical extent.

    The test is containment of the object's whole EXTENT, not of its anchor
    point. An object whose contact point falls just inside a frame but whose
    body is clipped by the edge is one the detector cannot fairly be expected
    to report; counting that view as a dissenting vote would punish a correct
    finding for the framing of an unrelated photo.
    """
    return sorted(idx for idx, poly in eroded.items() if poly.contains(probe))


# --------------------------------------------------------------------------- #
# Detection ingestion.
# --------------------------------------------------------------------------- #


def load_detections(payload: Any) -> dict:
    """Accept either a full `analyze_batch` payload or a bare results list;
    return {image_id: [issue, ...]} with boxes still normalized 0-1."""
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        raise ValueError("detections JSON must be an analyze_batch payload or a results list")
    out: dict = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        image_id = str(entry.get("image_id") or "")
        issues = entry.get("detected_issues") if isinstance(entry.get("detected_issues"), list) else []
        out[image_id] = [i for i in issues if isinstance(i, dict) and i.get("bbox")]
    return out


def load_products(payload: Any) -> dict:
    """{image_id: [{label, score, box}, ...]} — the localizer's `/detect`
    candidates, which the detector-first sweep already computes per photo."""
    if isinstance(payload, dict) and isinstance(payload.get("products"), dict):
        payload = payload["products"]
    if not isinstance(payload, dict):
        return {}
    out: dict = {}
    for image_id, cands in payload.items():
        if not isinstance(cands, list):
            continue
        rows = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            box = c.get("box") or c.get("bbox")
            if not isinstance(box, dict):
                continue
            rows.append(
                {
                    "label": str(c.get("label") or "object"),
                    "score": float(c.get("score") or 0.0),
                    "box": box,
                }
            )
        out[str(image_id)] = rows
    return out


def fetch_products_from_localizer(views: list, labels: list, cfg: MergeConfig) -> dict:
    """Call the same localizer service the detector-first sweep uses, so the
    product inventory is built from the identical candidate boxes rather than
    a second, differently-tuned detector."""
    import base64

    import httpx

    base = cfg.localizer_url.strip().rstrip("/")
    if not base:
        return {}
    out: dict = {}
    timeout = httpx.Timeout(cfg.localizer_timeout_ms / 1000.0)
    for view in views:
        ok, buf = cv2.imencode(".jpg", view.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            continue
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.post(f"{base}/detect", json={"image_data_url": data_url, "labels": labels})
            if res.status_code >= 400:
                print(f"[multiview] localizer {res.status_code} for {view.image_id}")
                continue
            cands = res.json().get("candidates") or []
        except Exception as error:  # noqa: BLE001
            print(f"[multiview] localizer unreachable for {view.image_id}: {error}")
            continue
        rows = []
        for c in cands:
            box = c.get("box")
            if isinstance(box, dict):
                rows.append(
                    {"label": str(c.get("label") or "object"), "score": float(c.get("score") or 0.0), "box": box}
                )
        out[view.image_id] = rows
    return out


# --------------------------------------------------------------------------- #
# Projection: normalized boxes -> canonical table coordinates.
# --------------------------------------------------------------------------- #


def box_pixels(box: dict, view: View) -> tuple:
    """Normalized {x,y,w,h} -> full-resolution pixel (x, y, w, h)."""
    return (
        float(box["x"]) * view.width,
        float(box["y"]) * view.height,
        float(box["w"]) * view.width,
        float(box["h"]) * view.height,
    )


def ground_point(box: dict, view: View, H: np.ndarray) -> tuple:
    """Project a box's table-contact point into canonical coordinates.

    The bottom-edge midpoint is used rather than the centroid because that is
    the point physically resting ON the plane the homography models, so it maps
    exactly from any viewpoint. A centroid sits half an object-height above the
    plane and shifts with viewing angle — enough, for tall boxed stock, to push
    the same product into two clusters.
    """
    x, y, w, h = box_pixels(box, view)
    pt = np.array([[x + w / 2.0, y + h]], dtype=np.float64)
    return tuple(project(H, pt)[0])


def canonical_quad(box: dict, view: View, H: np.ndarray):
    """The box's four corners projected into canonical coordinates, as a
    polygon. Used as an object's visual extent for the visibility test — for a
    tall object this reaches beyond its table footprint, which is correct:
    the question it answers is "was the whole thing in frame?", not "where
    does it touch?"."""
    x, y, w, h = box_pixels(box, view)
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
    return Polygon(project(H, corners)).buffer(0)


def canonical_width(box: dict, view: View, H: np.ndarray) -> float:
    """Width of a box's bottom edge after projection — the object's apparent
    footprint width in canonical units, used to scale clustering radii."""
    x, y, w, h = box_pixels(box, view)
    edge = np.array([[x, y + h], [x + w, y + h]], dtype=np.float64)
    p = project(H, edge)
    return float(np.linalg.norm(p[1] - p[0]))


# --------------------------------------------------------------------------- #
# Constrained clustering.
# --------------------------------------------------------------------------- #


class _Union:
    """Union-find that also tracks which views each cluster already contains,
    so a merge that would put two observations from the SAME photo into one
    cluster is refused. That constraint is what stops two genuinely adjacent
    identical phones from collapsing into a single product."""

    def __init__(self, views_of: list) -> None:
        self.parent = list(range(len(views_of)))
        self.images = [{v} for v in views_of]

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def can_union(self, i: int, j: int) -> bool:
        a, b = self.find(i), self.find(j)
        return a != b and not (self.images[a] & self.images[b])

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a == b:
            return
        self.parent[b] = a
        self.images[a] |= self.images[b]

    def groups(self) -> dict:
        out: dict = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def vertically_slack(poly, slack: float):
    """Grow a canonical box about its centre in Y only, leaving X untouched."""
    if slack <= 0:
        return poly
    return _affine_scale(poly, xfact=1.0, yfact=1.0 + slack, origin="center")


def _bounds_box(poly) -> Optional[dict]:
    """Axis-aligned {x,y,w,h} of a canonical polygon, in canvas pixels."""
    if poly is None or poly.is_empty:
        return None
    x0, y0, x1, y1 = poly.bounds
    return {"x": round(x0, 1), "y": round(y0, 1), "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)}


def x_overlap_iou(a, b) -> float:
    """1-D IoU of two canonical boxes along the fixture axis.

    The reliable half of a VLM box. Two photos of the same phone agree on X to
    a few pixels while disagreeing on Y by ~100, so this separates devices with
    far more margin than area overlap does.
    """
    ax0, _, ax1, _ = a.bounds
    bx0, _, bx1, _ = b.bounds
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    union = max(ax1, bx1) - min(ax0, bx0)
    return inter / union if union > 0 else 0.0


def _overlap_iou(a, b, slack: float = 0.0) -> float:
    """IoU of two canonical polygons, with optional vertical slack; 0 when they
    don't meet."""
    a, b = vertically_slack(a, slack), vertically_slack(b, slack)
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def cluster_geometries(geoms: list, view_ids: list, min_iou: float, slack: float = 0.0,
                       min_x_iou: float = 0.0) -> list:
    """Same constrained single-linkage, keyed on canonical box overlap.

    Used when a finding has no parent product to anchor it. Overlap beats
    centroid distance here because it is scale-free — no radius to guess. On
    real showroom photos the separation is emphatic: two views of one phone
    scored IoU 0.32 while every unrelated pair scored exactly 0.00.

    The reason distance struggles is that coarse VLM boxes jitter between
    photos by an amount comparable to the spacing between neighbouring
    products, so no single radius cleanly separates them. Area overlap does.
    """
    n = len(geoms)
    if n == 0:
        return []
    uf = _Union(view_ids)
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if view_ids[i] == view_ids[j]:
                continue
            x_iou = x_overlap_iou(geoms[i], geoms[j])
            if x_iou < min_x_iou:
                continue
            if _overlap_iou(geoms[i], geoms[j], slack) >= min_iou:
                # Rank by the along-fixture score — the confident axis.
                cands.append((-x_iou, i, j))
    for _, i, j in sorted(cands):
        if uf.can_union(i, j):
            uf.union(i, j)
    return [sorted(members) for members in uf.groups().values()]


def cluster_observations(points: list, view_ids: list, labels: list, radius: float, match_labels: bool) -> list:
    """Single-linkage clustering under the one-observation-per-view constraint.

    Pairs are considered in ascending distance order, so the tightest, most
    confident merges are committed before the marginal ones — and a marginal
    pair that would violate the per-view constraint is simply skipped rather
    than forcing a split later.
    """
    n = len(points)
    if n == 0:
        return []
    uf = _Union(view_ids)
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if view_ids[i] == view_ids[j]:
                continue  # same photo: cannot be the same physical object
            if match_labels and labels[i] and labels[j] and labels[i] != labels[j]:
                continue
            d = math.dist(points[i], points[j])
            if d <= radius:
                cands.append((d, i, j))
    for _, i, j in sorted(cands):
        if uf.can_union(i, j):
            uf.union(i, j)
    return [sorted(members) for members in uf.groups().values()]


# --------------------------------------------------------------------------- #
# The merge itself.
# --------------------------------------------------------------------------- #


def _iou_norm(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
    iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _containment(inner: dict, outer: dict) -> float:
    """Fraction of `inner` that lies inside `outer` — the right measure for
    attaching a small damage box to the large product box it sits on, where
    IoU would be near zero by construction."""
    ix1, iy1 = max(inner["x"], outer["x"]), max(inner["y"], outer["y"])
    ix2 = min(inner["x"] + inner["w"], outer["x"] + outer["w"])
    iy2 = min(inner["y"] + inner["h"], outer["y"] + outer["h"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = inner["w"] * inner["h"]
    return inter / area if area > 0 else 0.0


def attach_to_product(issue_box: dict, candidates: list) -> Optional[int]:
    """Index of the product candidate a damage box belongs to: the smallest
    candidate containing most of it. Smallest wins because product boxes nest
    (a phone inside a display stand inside a table region) and the tightest
    enclosing object is the one the damage is actually on."""
    best, best_area = None, float("inf")
    for i, cand in enumerate(candidates):
        if _containment(issue_box, cand["box"]) < 0.5:
            continue
        area = cand["box"]["w"] * cand["box"]["h"]
        if area < best_area:
            best, best_area = i, area
    return best


def _vote_views(seen: list, detected: set) -> list:
    """The electorate for a finding: every view that geometrically contains it,
    plus every view that actually reported it. A view that produced a detection
    manifestly saw the object, whatever the containment test concluded — so
    `visible_in` is always a superset of `detected_in`, and agreement can never
    exceed 1.0 through a geometry edge case."""
    return sorted(set(seen) | set(detected))


def merge(
    views: list,
    transforms: dict,
    polys: dict,
    detections: dict,
    products: dict,
    cfg: MergeConfig,
) -> dict:
    """Build the canonical product inventory and the deduplicated issue list."""
    aligned = [v for v in views if v.index in transforms]
    eroded = erode_footprints(polys, cfg.visibility_margin_px)

    # -- 1. product inventory ------------------------------------------- #
    obs_pts, obs_views, obs_labels, obs_meta, obs_quads, widths = [], [], [], [], [], []
    for view in aligned:
        H = transforms[view.index]
        for k, cand in enumerate(products.get(view.image_id, [])):
            obs_pts.append(ground_point(cand["box"], view, H))
            obs_views.append(view.index)
            obs_labels.append(cand["label"])
            obs_meta.append({"view": view.index, "cand": k})
            obs_quads.append(canonical_quad(cand["box"], view, H))
            widths.append(canonical_width(cand["box"], view, H))

    # Clustering radii are expressed relative to how wide a product actually is
    # in canonical units, so they hold whatever arbitrary scale the table plane
    # frame ended up with.
    median_width = float(np.median(widths)) if widths else 0.0
    product_radius = cfg.product_cluster_frac * median_width

    product_clusters = cluster_observations(
        obs_pts, obs_views, obs_labels, product_radius, cfg.require_label_match
    )
    product_of_obs: dict = {}
    product_extent: dict = {}
    product_rows = []
    for pid, members in enumerate(sorted(product_clusters, key=lambda m: (obs_pts[m[0]][1], obs_pts[m[0]][0]))):
        pts = [obs_pts[m] for m in members]
        anchor = (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])))
        for m in members:
            product_of_obs[m] = pid
        # Hull of every view's observation of this product: the largest extent
        # anyone saw. Requiring a view to contain THAT is the conservative
        # reading of "could this photo have reported the problem?".
        product_extent[pid] = unary_union([obs_quads[m] for m in members]).convex_hull
        seen = _vote_views(visible_in(product_extent[pid], eroded), {obs_views[m] for m in members})
        product_rows.append(
            {
                "product_id": pid,
                "label": max(set(obs_labels[m] for m in members), key=[obs_labels[m] for m in members].count),
                "canonical_xy": [round(anchor[0], 2), round(anchor[1], 2)],
                "detected_in": sorted({views[obs_views[m]].image_id for m in members}),
                "visible_in": [views[i].image_id for i in seen],
                "observations": [
                    {
                        "image_id": views[obs_meta[m]["view"]].image_id,
                        "box": products[views[obs_meta[m]["view"]].image_id][obs_meta[m]["cand"]]["box"],
                        "score": products[views[obs_meta[m]["view"]].image_id][obs_meta[m]["cand"]]["score"],
                    }
                    for m in members
                ],
            }
        )

    # Reverse index so a damage box can look up its product cluster.
    obs_by_view_cand = {(obs_meta[i]["view"], obs_meta[i]["cand"]): i for i in range(len(obs_meta))}

    # -- 2. damage observations ------------------------------------------ #
    d_pts, d_views, d_keys, d_meta, d_quads = [], [], [], [], []
    for view in aligned:
        H = transforms[view.index]
        cands = products.get(view.image_id, [])
        for k, issue in enumerate(detections.get(view.image_id, [])):
            box = issue.get("bbox")
            if not isinstance(box, dict):
                continue
            cand_idx = attach_to_product(box, cands) if cands else None
            pid = product_of_obs.get(obs_by_view_cand.get((view.index, cand_idx), -1)) if cand_idx is not None else None
            # Anchor on the parent product's contact point when we have one:
            # a scratch's own bottom edge floats above the table by however
            # high up the product it sits, whereas the product's does not.
            if cand_idx is not None:
                anchor = ground_point(cands[cand_idx]["box"], view, H)
            else:
                anchor = ground_point(box, view, H)
            d_pts.append(anchor)
            d_quads.append(canonical_quad(box, view, H))
            d_views.append(view.index)
            # Findings only ever merge within one issue type, and within one
            # product when the product is known. A cracked screen and a
            # missing price tag on the same handset stay separate findings.
            d_keys.append(f"{issue.get('issue_type_id')}|{pid if pid is not None else ''}")
            d_meta.append({"view": view.index, "issue": k, "product_id": pid, "anchor": anchor})

    # Cluster per (issue_type, product) key so the geometric radius never has
    # to arbitrate between different kinds of problem.
    groups = []
    for key in sorted(set(d_keys)):
        idxs = [i for i, k in enumerate(d_keys) if k == key]
        pid = d_meta[idxs[0]]["product_id"]
        if pid is not None:
            # Same product + same issue type = same problem, by definition.
            # Still split by view constraint via the clusterer for safety.
            sub = cluster_observations(
                [d_pts[i] for i in idxs], [d_views[i] for i in idxs], [""] * len(idxs), float("inf"), False
            )
        else:
            # No parent product: fall back to canonical box overlap.
            sub = cluster_geometries(
                [d_quads[i] for i in idxs], [d_views[i] for i in idxs],
                cfg.damage_iou_threshold, cfg.box_vertical_slack, cfg.damage_x_iou_threshold
            )
        for members in sub:
            groups.append([idxs[m] for m in members])

    # -- 3. visibility-aware voting -------------------------------------- #
    issue_rows = []
    for iid, members in enumerate(groups):
        pts = [d_pts[m] for m in members]
        anchor = (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])))
        detected_views = {d_views[m] for m in members}
        pid = d_meta[members[0]]["product_id"]
        # Vote over the parent product's extent when known — the damage box
        # itself is far too small a probe, and what actually determines whether
        # a photo could report a scratch is whether it framed the whole handset.
        # Without a parent product, the finding's own observed extent is the
        # best available stand-in for "how much of the scene had to be in
        # frame for this to be reportable".
        probe = (
            product_extent[pid]
            if pid is not None
            else unary_union([d_quads[m] for m in members]).convex_hull
        )
        seen = _vote_views(visible_in(probe, eroded), detected_views)
        detected_views = sorted(detected_views)
        # A view that saw the object but is not in `detected_views` is a genuine
        # dissent; a view that never saw it is silent, not dissenting.
        n_visible = len(seen)
        agreement = len(detected_views) / float(n_visible) if n_visible else 1.0

        raw = []
        for m in members:
            view = views[d_meta[m]["view"]]
            issue = detections[view.image_id][d_meta[m]["issue"]]
            raw.append((view, issue))
        best_view, best_issue = max(raw, key=lambda t: float(t[1].get("confidence") or 0.0))
        base_conf = float(best_issue.get("confidence") or 0.0)

        # With a single view there is no corroboration available in either
        # direction, so the detector's own confidence stands unmodified.
        # With several, agreement scales it between vote_floor and 1.0.
        merged_conf = base_conf if n_visible <= 1 else base_conf * (cfg.vote_floor + (1.0 - cfg.vote_floor) * agreement)

        issue_rows.append(
            {
                "issue_id": iid,
                "issue_type_id": best_issue.get("issue_type_id"),
                "product_id": pid,
                "object_label": best_issue.get("object_label"),
                "canonical_xy": [round(anchor[0], 2), round(anchor[1], 2)],
                "merged_confidence": round(max(0.0, min(1.0, merged_conf)), 4),
                "max_confidence": round(base_conf, 4),
                "support": {
                    "detected_in": [views[i].image_id for i in detected_views],
                    "visible_in": [views[i].image_id for i in seen],
                    "agreement": round(agreement, 3),
                },
                # Corroborated in every view that could see it — safe to trust.
                "corroborated": len(detected_views) > 1 and agreement >= 1.0,
                # At most half the views that could see it reported it: the
                # classic single-view VLM false positive. Not dropped — glare
                # and viewing angle genuinely hide real damage — but flagged.
                # `<=`, so a 1-of-2 split is flagged rather than passing as a
                # bare majority.
                "low_support": n_visible > 1 and agreement <= cfg.low_support_ratio,
                # The one box to draw for this finding: the highest-confidence
                # view's own normalized box, plus which photo it belongs to.
                # Deliberately NOT an average of the members — averaging boxes
                # that disagree vertically by ~100 px produces a box that fits
                # neither photo.
                "bbox": best_issue.get("bbox"),
                "bbox_image_id": best_view.image_id,
                # Same finding in the shared table frame, for drawing on the
                # canonical mosaic or comparing positions across findings.
                "canonical_bbox": _bounds_box(unary_union([d_quads[m] for m in members])),
                "best_view": best_view.image_id,
                "evidence": best_issue.get("evidence") or "",
                # Every view's wording, deduplicated — two photos of one defect
                # often describe it differently, and the extra phrasing is
                # evidence a reviewer wants, but the identical string repeated
                # once per view is not.
                "evidence_all": sorted({
                    (detections[views[d_meta[m]["view"]].image_id][d_meta[m]["issue"]].get("evidence") or "").strip()
                    for m in members
                } - {""}),
                "observations": [
                    {
                        "image_id": views[d_meta[m]["view"]].image_id,
                        "bbox": detections[views[d_meta[m]["view"]].image_id][d_meta[m]["issue"]].get("bbox"),
                        "confidence": detections[views[d_meta[m]["view"]].image_id][d_meta[m]["issue"]].get(
                            "confidence"
                        ),
                        "evidence": detections[views[d_meta[m]["view"]].image_id][d_meta[m]["issue"]].get(
                            "evidence"
                        ) or "",
                    }
                    for m in members
                ],
            }
        )

    below_floor = 0
    if cfg.min_confidence > 0:
        kept = {r["issue_id"] for r in issue_rows if r["merged_confidence"] >= cfg.min_confidence}
        # Counted separately from duplicates: a finding dropped for low
        # confidence was not a second sighting of something else.
        below_floor = sum(len(groups[i]) for i in range(len(groups)) if i not in kept)
        issue_rows = [r for r in issue_rows if r["issue_id"] in kept]
        groups = [g for i, g in enumerate(groups) if i in kept]
        for new_id, row in enumerate(issue_rows):
            row["issue_id"] = new_id

    issue_rows.sort(key=lambda r: -r["merged_confidence"])

    # -- 4. roll findings up to physical devices -------------------------- #
    # One handset can carry several findings at once — a cracked screen AND a
    # dark display are two issues on ONE device. Counting issue clusters would
    # report that phone twice, so devices are grouped separately: by product
    # when the localizer identified one, by canonical box overlap otherwise.
    # The one-per-view constraint used elsewhere is deliberately NOT applied —
    # two findings in the same photo can, and often do, share a device.
    group_extent = [unary_union([d_quads[m] for m in members]).convex_hull for members in groups]
    group_views = [{d_views[m] for m in members} for members in groups]
    parent = list(range(len(groups)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            pi, pj = d_meta[groups[i][0]]["product_id"], d_meta[groups[j][0]]["product_id"]
            # Vertical slack exists to absorb CROSS-view box error. Two findings
            # that share a photo have no such error between them — the same
            # model drew both boxes on the same pixels — so they are compared
            # strictly. Without this, slack sized for cross-view jitter also
            # merges objects at different depths: a phone on the table and a
            # laptop on the counter behind it grouped as one device.
            slack = 0.0 if (group_views[i] & group_views[j]) else cfg.box_vertical_slack
            same = (
                pi is not None and pi == pj
                if pi is not None or pj is not None
                else (
                    x_overlap_iou(group_extent[i], group_extent[j]) >= cfg.damage_x_iou_threshold
                    and _overlap_iou(group_extent[i], group_extent[j], slack)
                    >= cfg.device_iou_threshold
                )
            )
            if same:
                parent[find(i)] = find(j)

    by_device: dict = {}
    for idx in range(len(groups)):
        by_device.setdefault(find(idx), []).append(idx)

    device_rows = []
    for dev_id, members in enumerate(sorted(by_device.values(), key=lambda m: -max(
            issue_rows[[r["issue_id"] for r in issue_rows].index(i)]["merged_confidence"] for i in m))):
        issues = [r for r in issue_rows if r["issue_id"] in members]
        detected = sorted({img for r in issues for img in r["support"]["detected_in"]})
        for r in issues:
            r["device_id"] = dev_id
        device_rows.append(
            {
                "device_id": dev_id,
                "object_label": next((r["object_label"] for r in issues if r.get("object_label")), None),
                "product_id": d_meta[groups[members[0]][0]]["product_id"],
                "canonical_xy": issues[0]["canonical_xy"],
                "canonical_bbox": _bounds_box(unary_union([group_extent[i] for i in members])),
                "confidence": max(r["merged_confidence"] for r in issues),
                "detected_in": detected,
                "issues": [
                    {"issue_id": r["issue_id"], "issue_type_id": r["issue_type_id"],
                     "confidence": r["merged_confidence"], "bbox_count": len(r["observations"])}
                    for r in issues
                ],
            }
        )

    raw_issue_count = sum(len(detections.get(v.image_id, [])) for v in aligned)
    raw_product_count = len(obs_pts)
    return {
        "devices": device_rows,
        "products": product_rows,
        "issues": issue_rows,
        "counts": {
            "distinct_devices": len(device_rows),
            "distinct_products": len(product_rows),
            "product_observations": raw_product_count,
            "distinct_issues": len(issue_rows),
            "issue_observations": raw_issue_count,
            "duplicates_removed": raw_issue_count - len(issue_rows) - below_floor,
            "below_confidence_floor": below_floor,
            "low_support_issues": sum(1 for r in issue_rows if r["low_support"]),
        },
        "scale": {"median_product_width_px": round(median_width, 2), "product_radius_px": round(product_radius, 2)},
    }


# --------------------------------------------------------------------------- #
# Debug visualization — a canonical mosaic with footprints, products, issues.
# --------------------------------------------------------------------------- #


_PALETTE = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
]


def render_canvas(views: list, transforms: dict, polys: dict, merged: dict, size: tuple) -> np.ndarray:
    """Warp every aligned photo into the canonical frame and overlay the merge.

    The single most useful debugging artifact here: if products are ghosted or
    doubled in the mosaic the alignment is wrong, and no amount of threshold
    tuning downstream will fix the counts.
    """
    W, H = size
    canvas = np.zeros((H, W, 3), np.uint8)
    accum = np.zeros((H, W), np.float32)
    for idx, M in sorted(transforms.items()):
        warped = cv2.warpPerspective(views[idx].bgr, M, (W, H))
        mask = cv2.warpPerspective(np.ones(views[idx].bgr.shape[:2], np.float32), M, (W, H))
        # Running average, so overlap regions blend instead of the last view
        # simply painting over the others — ghosting is the misalignment tell.
        weight = np.clip(mask, 0, 1)
        total = accum + weight
        safe = np.where(total > 0, total, 1.0)[..., None]
        blended = canvas.astype(np.float32) * accum[..., None] + warped.astype(np.float32) * weight[..., None]
        canvas = (blended / safe).astype(np.uint8)
        accum = total

    overlay = canvas.copy()
    for n, (idx, poly) in enumerate(sorted(polys.items())):
        color = _PALETTE[n % len(_PALETTE)]
        pts = np.array(poly.exterior.coords, np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, color, 3)
        cx, cy = np.array(poly.centroid.coords)[0]
        cv2.putText(overlay, views[idx].image_id, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    for row in merged["products"]:
        x, y = int(row["canonical_xy"][0]), int(row["canonical_xy"][1])
        cv2.circle(overlay, (x, y), 9, (255, 255, 255), -1)
        cv2.circle(overlay, (x, y), 9, (40, 40, 40), 2)
        cv2.putText(overlay, f"P{row['product_id']}", (x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    for row in merged["issues"]:
        x, y = int(row["canonical_xy"][0]), int(row["canonical_xy"][1])
        color = (0, 200, 0) if row["corroborated"] else ((0, 165, 255) if row["low_support"] else (0, 0, 255))
        cv2.drawMarker(overlay, (x, y), color, cv2.MARKER_TILTED_CROSS, 26, 3)
        label = f"{row['issue_type_id']} {row['merged_confidence']:.2f}"
        cv2.putText(overlay, label, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0)


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def spatial_order(transforms: dict, polys: dict) -> list:
    """Order the views along the table, from their own geometry.

    Derived from where each footprint actually landed rather than from
    filenames or EXIF timestamps: the operator walks the length of the fixture,
    so the principal axis of the footprint centroids IS the walk. This makes
    "adjacent photos" a spatial fact, which is what overlap has to be measured
    between, and it holds even when the files arrive out of order.
    """
    idxs = sorted(i for i in transforms if i in polys)
    if len(idxs) < 2:
        return idxs
    pts = np.array([np.array(polys[i].centroid.coords)[0] for i in idxs])
    centred = pts - pts.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    return [idxs[k] for k in np.argsort(centred @ axis)]


def capture_quality(views: list, transforms: dict, polys: dict, pairs: list, cfg: MergeConfig) -> dict:
    """Grade the capture itself, in terms the person holding the phone can act on.

    The rule being enforced: photo i and photo i+2 must share ground, because
    that third edge is what closes a triangle and lets the alignment check
    itself. That happens only above 50% consecutive overlap — at exactly 50%,
    frames i and i+2 abut with zero shared area. Measured on a synthetic chain,
    triangles first appear between 50% and 60%, so the default asks for a
    margin above the cliff rather than sitting on it.

    Below that, a chain capture has no redundancy anywhere and an aliased
    homography is undetectable rather than merely unchecked.
    """
    order = spatial_order(transforms, polys)
    gaps = []
    for a, b in zip(order, order[1:]):
        inter = polys[a].intersection(polys[b])
        area = inter.area if not inter.is_empty else 0.0
        frac = max(
            area / polys[a].area if polys[a].area else 0.0,
            area / polys[b].area if polys[b].area else 0.0,
        )
        gaps.append(
            {
                "images": [views[a].image_id, views[b].image_id],
                "overlap": round(frac, 3),
                "sufficient": bool(frac >= cfg.min_consecutive_overlap),
            }
        )

    unverified = unverified_edges(pairs)
    weak = [g for g in gaps if not g["sufficient"]]
    advice = []
    for g in weak:
        advice.append(
            f"Only {g['overlap']:.0%} overlap between {g['images'][0]} and {g['images'][1]} — "
            f"take an extra photo between them (aim for ~{cfg.target_consecutive_overlap:.0%})."
        )
    if unverified and not weak:
        # Enough overlap pairwise, but the graph still has a bare edge — e.g. a
        # photo of one end that nothing else reaches around to.
        advice.append(
            "Some photos are linked by a single unverified edge. One wide shot of the whole "
            "fixture would tie everything together."
        )
    return {
        "order": [views[i].image_id for i in order],
        "consecutive_gaps": gaps,
        "weakest_overlap": round(min((g["overlap"] for g in gaps), default=0.0), 3),
        "unverified_edges": [[views[a].image_id, views[b].image_id] for a, b in sorted(unverified)],
        "self_verifying": not unverified,
        "verdict": "ok" if not weak and not unverified else ("sparse" if weak else "unverified"),
        "advice": advice,
    }


def align(
    image_paths: list,
    *,
    image_ids: Optional[list] = None,
    cfg: Optional[MergeConfig] = None,
    verbose: bool = True,
) -> dict:
    """Match, verify and globally align the photos. No detections needed.

    Split out from `run` so the capture can be graded BEFORE any VLM is
    invoked: matching is cheap, damage detection is not, and re-shooting is
    only possible while the operator is still standing at the fixture.
    """
    cfg = cfg or MergeConfig()
    views = load_views(image_paths, image_ids)
    if len(views) < 2:
        raise ValueError("multi-view merge needs at least 2 images")

    matcher = FeatureMatcher(cfg)
    if verbose:
        print(f"[multiview] backend={matcher.backend} device={cfg.resolved_device()} views={len(views)}")

    pairs = build_graph(views, matcher, cfg, verbose)
    if not any(p.accepted for p in pairs):
        raise RuntimeError(
            "no image pair produced a trustworthy homography — the photos may not overlap, "
            "or the surface is too reflective/repetitive for feature matching. Try --extractor disk, "
            "or capture with more overlap."
        )

    if cfg.cycle_check:
        pairs = prune_inconsistent_cycles(pairs, views, cfg, verbose)
        if not any(p.accepted for p in pairs):
            raise RuntimeError("every image pair failed loop-closure consistency — alignment is not trustworthy")

    ref = choose_reference(pairs, len(views))
    transforms = spanning_transforms(pairs, len(views), ref)
    if cfg.refine_global:
        transforms = refine_global(transforms, pairs, views, ref, cfg)
    transforms, size = normalize_canvas(transforms, views, cfg)

    dropped = [v.image_id for v in views if v.index not in transforms]
    if dropped and verbose:
        # These photos never connected to the graph. Their detections are
        # excluded from the merged counts — surfacing that is essential, since
        # silently dropping a photo would understate the inventory.
        print(f"[multiview] WARNING: unaligned, excluded from merge: {', '.join(dropped)}")

    unverified = unverified_edges(pairs)
    if unverified and verbose:
        names = ", ".join(f"{views[a].image_id}-{views[b].image_id}" for a, b in sorted(unverified))
        print(
            f"[multiview] WARNING: {len(unverified)} edge(s) belong to no triangle and are "
            f"UNVERIFIED: {names}. Nothing corroborates them, so an alias there cannot be "
            f"detected. Add a wide overview shot, or more overlap between non-adjacent photos."
        )

    rms = alignment_rms(transforms, pairs)
    # Scale-free: the canonical frame is normalized to `canvas_max_side`, so a
    # percentage of that is comparable across jobs regardless of photo size.
    rms_frac = rms / float(max(size)) if size and max(size) else float("nan")
    if verbose and np.isfinite(rms):
        note = "" if rms_frac < 0.01 else "  <-- HIGH, treat counts as unreliable"
        print(f"[multiview] alignment RMS {rms:.2f} px ({rms_frac:.2%} of canvas){note}")

    polys = footprints(transforms, views)
    quality = capture_quality(views, transforms, polys, pairs, cfg)
    if verbose:
        print(f"[multiview] capture: {quality['verdict']} (weakest overlap {quality['weakest_overlap']:.0%})")
        for line in quality["advice"]:
            print(f"[multiview]   -> {line}")

    return {
        "views": views,
        "transforms": transforms,
        "polys": polys,
        "pairs": pairs,
        "size": size,
        "ref": ref,
        "rms": rms,
        "rms_frac": rms_frac,
        "dropped": dropped,
        "unverified": unverified,
        "backend": matcher.backend,
        "capture": quality,
    }


def check_capture(
    image_paths: list,
    *,
    image_ids: Optional[list] = None,
    cfg: Optional[MergeConfig] = None,
    verbose: bool = True,
) -> dict:
    """Pre-flight: is this set of photos good enough to merge? Returns
    `capture_quality`'s verdict without touching detections or any LLM."""
    return align(image_paths, image_ids=image_ids, cfg=cfg, verbose=verbose)["capture"]


def run(
    image_paths: list,
    detections_payload: Any,
    *,
    products_payload: Any = None,
    image_ids: Optional[list] = None,
    cfg: Optional[MergeConfig] = None,
    detector_labels: Optional[list] = None,
    verbose: bool = True,
) -> dict:
    """Full pipeline. Returns the merged report plus the geometry that produced
    it, so a caller can re-render or re-threshold without re-matching."""
    cfg = cfg or MergeConfig()
    geo = align(image_paths, image_ids=image_ids, cfg=cfg, verbose=verbose)
    views, transforms, polys = geo["views"], geo["transforms"], geo["polys"]
    pairs, size, ref = geo["pairs"], geo["size"], geo["ref"]
    rms, rms_frac, dropped, unverified = geo["rms"], geo["rms_frac"], geo["dropped"], geo["unverified"]

    detections = load_detections(detections_payload)

    if products_payload is not None:
        products = load_products(products_payload)
    elif cfg.localizer_url:
        labels = detector_labels or ["product", "phone", "box", "device"]
        products = fetch_products_from_localizer([v for v in views if v.index in transforms], labels, cfg)
    else:
        products = {}
    if not products and verbose:
        print(
            "[multiview] no product candidates supplied — product counting is disabled and damage "
            "findings will be clustered on their own geometry (less reliable). Pass --products or "
            "--localizer-url."
        )

    merged = merge(views, transforms, polys, detections, products, cfg)

    overlaps = overlap_regions(polys)
    merged["alignment"] = {
        "reference_image_id": views[ref].image_id,
        "backend": geo["backend"],
        "aligned": [views[i].image_id for i in sorted(transforms)],
        "unaligned": dropped,
        "canonical_size": list(size),
        "rms_px": None if not np.isfinite(rms) else round(rms, 3),
        "rms_fraction_of_canvas": None if not np.isfinite(rms_frac) else round(rms_frac, 5),
        "trustworthy": bool(np.isfinite(rms) and rms_frac < 0.01),
        "unverified_edges": [[views[a].image_id, views[b].image_id] for a, b in sorted(unverified)],
        "capture": geo["capture"],
        "transforms": {views[i].image_id: transforms[i].tolist() for i in sorted(transforms)},
        "pairs": [p.to_json() for p in pairs],
        "overlaps": [
            {
                "images": [views[a].image_id, views[b].image_id],
                "area_px": round(poly.area, 1),
                "fraction_of_a": round(poly.area / polys[a].area, 3) if polys[a].area else 0.0,
                "fraction_of_b": round(poly.area / polys[b].area, 3) if polys[b].area else 0.0,
                "polygon": [[round(x, 1), round(y, 1)] for x, y in np.array(poly.exterior.coords)]
                if poly.geom_type == "Polygon"
                else [],
            }
            for (a, b), poly in sorted(overlaps.items())
        ],
    }
    merged["_geometry"] = {"views": views, "transforms": transforms, "polys": polys, "size": size}
    return merged


def _json_safe(report: dict) -> dict:
    return {k: v for k, v in report.items() if not k.startswith("_")}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory or glob of the photos of ONE fixture")
    ap.add_argument("--detections", help="analyze_batch JSON payload (not needed with --check)")
    ap.add_argument("--check", action="store_true",
                    help="pre-flight only: grade the capture and exit, before spending any VLM tokens")
    ap.add_argument("--products", help="localizer /detect candidates as {image_id: [...]}")
    ap.add_argument("--localizer-url", default=os.getenv("LOCALIZER_URL", ""), help="fetch products live instead")
    ap.add_argument("--out", default="multiview_out", help="output directory")
    ap.add_argument("--viz", action="store_true", help="render the canonical mosaic")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop merged findings below this confidence (applied after cross-view voting)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-image diagnostics; print only the final merged numbers")
    ap.add_argument("--extractor", default="superpoint", choices=["superpoint", "disk", "aliked", "sift"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--min-inliers", type=int, default=MergeConfig.min_inliers)
    ap.add_argument("--min-inlier-ratio", type=float, default=MergeConfig.min_inlier_ratio)
    ap.add_argument("--no-refine", action="store_true", help="skip global least-squares refinement")
    args = ap.parse_args(argv)

    if os.path.isdir(args.images):
        paths = sorted(
            p for p in glob.glob(os.path.join(args.images, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
    else:
        paths = sorted(glob.glob(args.images))
    if len(paths) < 2:
        print(f"need at least 2 images, found {len(paths)}", file=sys.stderr)
        return 2

    cfg_check = MergeConfig(extractor=args.extractor, device=args.device, min_inliers=args.min_inliers,
                            min_inlier_ratio=args.min_inlier_ratio, refine_global=not args.no_refine,
                            localizer_url=args.localizer_url, min_confidence=args.min_confidence)
    if args.check:
        quality = check_capture(paths, cfg=cfg_check)
        print()
        for gap in quality["consecutive_gaps"]:
            mark = "ok " if gap["sufficient"] else "LOW"
            print(f"  {mark} {gap['images'][0]} -> {gap['images'][1]}: {gap['overlap']:.0%} overlap")
        # `align` already printed the advice lines; don't repeat them here.
        print(f"\n  verdict: {quality['verdict']}  (self-verifying: {quality['self_verifying']})")
        # Non-zero exit so a capture app can gate on this and prompt a re-shoot
        # while the operator is still standing at the fixture.
        return 0 if quality["verdict"] == "ok" else 1

    if not args.detections:
        print("--detections is required unless --check is given", file=sys.stderr)
        return 2
    with open(args.detections) as fh:
        detections_payload = json.load(fh)
    products_payload = None
    if args.products:
        with open(args.products) as fh:
            products_payload = json.load(fh)

    cfg = MergeConfig(
        extractor=args.extractor,
        device=args.device,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        refine_global=not args.no_refine,
        localizer_url=args.localizer_url,
        min_confidence=args.min_confidence,
    )
    report = run(paths, detections_payload, products_payload=products_payload, cfg=cfg,
                 verbose=not args.quiet)

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "merged.json")
    with open(out_json, "w") as fh:
        json.dump(_json_safe(report), fh, indent=2)

    c = report["counts"]
    print()
    print(f"  devices with issues : {c['distinct_devices']}")
    print(f"  bounding boxes      : {c['distinct_issues']}")
    dropped = (f", {c['below_confidence_floor']} below confidence floor"
               if c.get("below_confidence_floor") else "")
    print(f"  raw detections      : {c['issue_observations']}"
          f"  ({c['duplicates_removed']} duplicate{'' if c['duplicates_removed'] == 1 else 's'} merged{dropped})")
    if c["distinct_products"]:
        print(f"  products on fixture : {c['distinct_products']}")
    for row in report["devices"]:
        types = ", ".join(sorted({i["issue_type_id"] for i in row["issues"]}))
        print(f"    device {row['device_id']}: {types}  (conf {row['confidence']:.2f}, "
              f"seen in {len(row['detected_in'])} photo(s))")
    print(f"\n  wrote {out_json}")

    if args.viz:
        g = report["_geometry"]
        canvas = render_canvas(g["views"], g["transforms"], g["polys"], report, g["size"])
        out_png = os.path.join(args.out, "canonical.png")
        cv2.imwrite(out_png, canvas)
        print(f"[multiview] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
