"""MuseTalk V1.5 realtime streaming engine.

Loads the MuseTalk models ONCE, prepares an avatar from a single photo ONCE
(cached to disk), then exposes a *streaming* frame generator: given an audio
clip it yields blended BGR frames one batch at a time, so the caller can push
them to the client as they are produced instead of rendering the whole video
first.

This runs inside the isolated MuseTalk venv (torch 2.0.1 + mmpose). It must run
with the MuseTalk repo as CWD because MuseTalk's preprocessing.py loads dwpose
with hard-coded relative paths at import time.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

MUSETALK_ROOT = os.environ.get(
    "MUSETALK_ROOT", "/home/pragay/WWAI/STT_TTS_experimentation/MuseTalk"
)

# MuseTalk's preprocessing.py initializes dwpose/face-detection at import time
# using paths relative to the repo root, so we must be there before importing.
os.chdir(MUSETALK_ROOT)
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

import cv2  # noqa: E402
import torch  # noqa: E402
from transformers import WhisperModel  # noqa: E402

from musetalk.utils.utils import load_all_model, datagen  # noqa: E402
from musetalk.utils.blending import (  # noqa: E402
    get_image_prepare_material,
    get_image_blending,
)
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402

_COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)


@dataclass
class Avatar:
    avatar_id: str
    frame_list_cycle: list
    coord_list_cycle: list
    latent_cycle: list
    mask_list_cycle: list
    mask_coords_list_cycle: list
    frame_shape: tuple  # (h, w)


class MuseTalkEngine:
    def __init__(
        self,
        version: str = "v15",
        gpu_id: int = 0,
        batch_size: int = 4,
        fps: int = 25,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        audio_padding_left: int = 2,
        audio_padding_right: int = 2,
        avatar_max_dim: int = 768,
        avatar_max_seconds: float = 6.0,
    ) -> None:
        self.avatar_max_dim = int(avatar_max_dim)
        # Cap the base clip by DURATION and sample it at the render fps, so head
        # motion plays at natural real-time speed (not temporally compressed).
        self.avatar_max_seconds = float(avatar_max_seconds)
        self.version = version
        self.fps = int(fps)
        self.batch_size = int(batch_size)
        self.extra_margin = int(extra_margin)
        self.parsing_mode = parsing_mode
        self.audio_padding_left = int(audio_padding_left)
        self.audio_padding_right = int(audio_padding_right)
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        unet_dir = "./models/musetalkV15" if version == "v15" else "./models/musetalk"
        unet_model_path = (
            f"{unet_dir}/unet.pth" if version == "v15" else f"{unet_dir}/pytorch_model.bin"
        )
        unet_config = f"{unet_dir}/musetalk.json"

        t0 = time.time()
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type="sd-vae",
            unet_config=unet_config,
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.weight_dtype = self.unet.model.dtype

        self.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        self.whisper = (
            WhisperModel.from_pretrained("./models/whisper")
            .to(device=self.device, dtype=self.weight_dtype)
            .eval()
        )
        self.whisper.requires_grad_(False)

        # FaceParsing + (later) dwpose/face-align are only needed to *build* an
        # avatar. Create them lazily so a cached-avatar startup keeps just
        # UNet+VAE+Whisper resident (~2 GB) and fits next to Chatterbox on 6 GB.
        self.fp = None
        self._fp_cheeks = (left_cheek_width, right_cheek_width)

        print(f"[engine] models loaded in {time.time() - t0:.1f}s on {self.device}", flush=True)

    def _ensure_face_parser(self):
        if self.fp is None:
            if self.version == "v15":
                self.fp = FaceParsing(
                    left_cheek_width=self._fp_cheeks[0], right_cheek_width=self._fp_cheeks[1]
                )
            else:
                self.fp = FaceParsing()

    # ------------------------------------------------------------------
    # Avatar preparation (runs once per photo; cached to disk)
    # ------------------------------------------------------------------
    def prepare_avatar(
        self,
        image_path: str,
        avatar_id: str = "default",
        bbox_shift: int = 0,
        cache_dir: str = "./results/realtime_avatars",
        force: bool = False,
    ) -> Avatar:
        adir = os.path.join(cache_dir, f"{self.version}_{avatar_id}")
        latents_p = os.path.join(adir, "latents.pt")
        coords_p = os.path.join(adir, "coords.pkl")
        maskcoords_p = os.path.join(adir, "mask_coords.pkl")
        frames_dir = os.path.join(adir, "frames")
        masks_dir = os.path.join(adir, "masks")
        meta_p = os.path.join(adir, "meta.json")

        cached = all(
            os.path.exists(p) for p in (latents_p, coords_p, maskcoords_p, meta_p)
        ) and os.path.isdir(frames_dir)

        if cached and not force:
            print(f"[engine] loading cached avatar '{avatar_id}'", flush=True)
            latent_cycle = torch.load(latents_p)
            with open(coords_p, "rb") as f:
                coord_list_cycle = pickle.load(f)
            with open(maskcoords_p, "rb") as f:
                mask_coords_list_cycle = pickle.load(f)
            frame_list_cycle = self._read_imgs_sorted(frames_dir)
            mask_list_cycle = self._read_imgs_sorted(masks_dir, grayscale=True)
            with open(meta_p) as f:
                meta = json.load(f)
            self.avatar = Avatar(
                avatar_id, frame_list_cycle, coord_list_cycle, latent_cycle,
                mask_list_cycle, mask_coords_list_cycle, tuple(meta["frame_shape"]),
            )
            return self.avatar

        # Build from scratch. Import preprocessing lazily so engine import doesn't
        # require dwpose/face weights until an avatar is actually prepared.
        from musetalk.utils.preprocessing import get_landmark_and_bbox

        self._ensure_face_parser()
        print(f"[engine] preparing avatar '{avatar_id}' from {image_path}", flush=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        # Source can be a single photo OR an idle video. A video keeps the head's
        # natural motion/blinks under the repainted mouth, which is what makes the
        # avatar read as actually speaking instead of a frozen puppet.
        src_frame_paths = self._extract_source_frames(image_path, adir)

        coord_list, frame_list = get_landmark_and_bbox(src_frame_paths, bbox_shift)

        # Keep only frames with a detected face; for video this drops occasional bad
        # frames instead of failing the whole avatar.
        kept_frames: list = []
        kept_coords: list = []
        latent_list: list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == _COORD_PLACEHOLDER:
                continue
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = min(y2 + self.extra_margin, frame.shape[0])
            bbox = [x1, y1, x2, y2]
            crop = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latent_list.append(self.vae.get_latents_for_unet(resized))
            kept_frames.append(frame)
            kept_coords.append(bbox)

        if not kept_frames:
            raise RuntimeError(
                "No face detected in the avatar source. Use a clear, front-facing photo/video."
            )

        # Ping-pong loop (forward then backward) so a short idle clip repeats with no
        # visible jump-cut during long utterances.
        frame_list_cycle = kept_frames + kept_frames[::-1]
        coord_list_cycle = kept_coords + kept_coords[::-1]
        latent_cycle = latent_list + latent_list[::-1]

        mask_list_cycle: list = []
        mask_coords_list_cycle: list = []
        mode = self.parsing_mode if self.version == "v15" else "raw"
        for i, frame in enumerate(frame_list_cycle):
            x1, y1, x2, y2 = coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2], fp=self.fp, mode=mode
            )
            mask_list_cycle.append(mask)
            mask_coords_list_cycle.append(crop_box)
            cv2.imwrite(os.path.join(frames_dir, f"{i:08d}.png"), frame)
            cv2.imwrite(os.path.join(masks_dir, f"{i:08d}.png"), mask)

        torch.save(latent_cycle, latents_p)
        with open(coords_p, "wb") as f:
            pickle.dump(coord_list_cycle, f)
        with open(maskcoords_p, "wb") as f:
            pickle.dump(mask_coords_list_cycle, f)
        with open(meta_p, "w") as f:
            json.dump({"frame_shape": list(kept_frames[0].shape[:2]), "bbox_shift": bbox_shift}, f)

        self.avatar = Avatar(
            avatar_id, frame_list_cycle, coord_list_cycle, latent_cycle,
            mask_list_cycle, mask_coords_list_cycle, tuple(kept_frames[0].shape[:2]),
        )
        print(f"[engine] avatar ready ({len(kept_frames)} base frame(s), "
              f"{'video' if len(kept_frames) > 1 else 'photo'})", flush=True)
        return self.avatar

    _VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

    def _resize_to_max(self, im):
        m = max(im.shape[:2])
        if self.avatar_max_dim and m > self.avatar_max_dim:
            s = self.avatar_max_dim / m
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        return im

    def _extract_source_frames(self, src: str, adir: str) -> list:
        """Return a list of (downscaled) frame image paths for a photo or video."""
        raw_dir = os.path.join(adir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ext = os.path.splitext(src)[1].lower()

        if ext in self._VIDEO_EXTS:
            cap = cv2.VideoCapture(src)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.fps)
            frames = []
            while True:
                ret, fr = cap.read()
                if not ret:
                    break
                frames.append(fr)
            cap.release()
            if not frames:
                raise RuntimeError(f"Could not read frames from video: {src}")
            total = len(frames)
            # Resample to the render fps and cap by duration: pick source frame
            # round(i * src_fps / render_fps) for i in [0, n). This preserves the
            # real-time pace of the motion when the cycle is advanced at render fps.
            max_n = int(self.avatar_max_seconds * self.fps)
            avail_n = int(round(total * self.fps / max(src_fps, 1e-6)))
            n_target = max(1, min(max_n, avail_n))
            sel = [min(total - 1, int(round(i * src_fps / self.fps))) for i in range(n_target)]
            paths = []
            for i, si in enumerate(sel):
                p = os.path.join(raw_dir, f"{i:06d}.png")
                cv2.imwrite(p, self._resize_to_max(frames[si]))
                paths.append(p)
            print(f"[engine] sampled {len(paths)} frames @ {self.fps}fps "
                  f"from {total}f/{src_fps:.0f}fps video ({self.avatar_max_seconds}s cap)", flush=True)
            return paths

        im = cv2.imread(src)
        if im is None:
            raise RuntimeError(f"Could not read avatar image: {src}")
        p = os.path.join(raw_dir, "000000.png")
        cv2.imwrite(p, self._resize_to_max(im))
        return [p]

    @staticmethod
    def _read_imgs_sorted(d: str, grayscale: bool = False) -> list:
        paths = sorted(
            glob.glob(os.path.join(d, "*.png")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        return [cv2.imread(p, flag) for p in paths]

    # ------------------------------------------------------------------
    # Audio -> whisper feature windows (one per output video frame)
    # ------------------------------------------------------------------
    def whisper_chunks_for_audio(self, audio_path: str):
        feats, librosa_length = self.audio_processor.get_audio_feature(
            audio_path, weight_dtype=self.weight_dtype
        )
        chunks = self.audio_processor.get_whisper_chunk(
            feats,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=self.fps,
            audio_padding_length_left=self.audio_padding_left,
            audio_padding_length_right=self.audio_padding_right,
        )
        return chunks

    # ------------------------------------------------------------------
    # Streaming frame generator. Yields blended BGR frames one at a time,
    # emitting each batch's frames as soon as the VAE decodes them.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def stream_frames(self, audio_path: str, start_idx: int = 0) -> Iterator[np.ndarray]:
        if self.avatar is None:
            raise RuntimeError("Call prepare_avatar() before stream_frames().")
        av = self.avatar
        n_cycle = len(av.coord_list_cycle)

        whisper_chunks = self.whisper_chunks_for_audio(audio_path)
        gen = datagen(whisper_chunks, av.latent_cycle, self.batch_size)

        idx = start_idx
        for whisper_batch, latent_batch in gen:
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
            pred_latents = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                bbox = av.coord_list_cycle[idx % n_cycle]
                ori_frame = av.frame_list_cycle[idx % n_cycle].copy()
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    idx += 1
                    continue
                mask = av.mask_list_cycle[idx % n_cycle]
                mask_crop_box = av.mask_coords_list_cycle[idx % n_cycle]
                combined = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                yield combined
                idx += 1

    @torch.no_grad()
    def warmup(self) -> None:
        """Run one dummy UNet+VAE batch so cuDNN autotune happens at startup, not
        on the first real sentence (which otherwise adds several seconds)."""
        if self.avatar is None:
            return
        bs = max(1, self.batch_size)
        dummy_whisper = torch.zeros((bs, 50, 384), dtype=self.weight_dtype, device=self.device)
        latent = self.avatar.latent_cycle[0].to(self.device, self.unet.model.dtype)
        latent_batch = latent.repeat(bs, 1, 1, 1)
        af = self.pe(dummy_whisper)
        pred = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=af).sample
        pred = pred.to(device=self.device, dtype=self.vae.vae.dtype)
        _ = self.vae.decode_latents(pred)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[engine] warmup done", flush=True)

    def idle_frame(self) -> np.ndarray:
        """Return the avatar's neutral frame (for when nothing is being spoken)."""
        if self.avatar is None:
            raise RuntimeError("Call prepare_avatar() first.")
        return self.avatar.frame_list_cycle[0].copy()
