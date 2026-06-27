"""Quick comparison runner.

This script runs one or more model scripts as subprocesses so you get
consistent JSON-ish metric output lines.

Example:
  python tts_quick_compare.py --text "Hello" --models vits deepgram

Note: Cloud scripts require API keys in .env/env.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from typing import List, Optional

import tts_common as common


logger = logging.getLogger(__name__)


MODEL_TO_SCRIPT = {
    "vits": "tts_vits_coqui.py",
    "melotts": "tts_melotts.py",
    "deepgram": "tts_deepgram_aura2.py",
    "cartesia": "tts_cartesia_sonic3.py",
    "voxtream": "tts_voxtream.py",
    "vibevoice": "tts_vibevoice.py",
    "fishspeech": "tts_fishspeech_15.py",
    "cosyvoice2": "tts_cosyvoice2.py",
    "omnitalker": "tts_omnitalker.py",
    "chatterbox": "tts_chatterbox.py",
}


def main(argv: Optional[List[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument(
        "--models",
        nargs="+",
        default=["vits"],
        choices=sorted(MODEL_TO_SCRIPT.keys()),
    )
    p.add_argument("--no-play", action="store_true", help="Pass --no-play to each runner")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--stream-local", action="store_true", help="Pass --stream to local runners (TTFT comparisons)")
    p.add_argument(
        "--voxtream-prompt-audio",
        type=str,
        default="",
        help="Prompt WAV path to pass to VoXtream runner (optional).",
    )
    p.add_argument("--gpu", action="store_true", help="Force GPU for local runners")
    p.add_argument("--cpu", action="store_true", help="Force CPU for local runners")
    args = p.parse_args(argv)

    py = sys.executable
    root = os.path.dirname(os.path.abspath(__file__))

    for m in args.models:
        script = os.path.join(root, MODEL_TO_SCRIPT[m])
        cmd = [py, script, "--text", args.text]

        if m == "voxtream" and args.voxtream_prompt_audio:
            cmd.extend(["--prompt-audio", args.voxtream_prompt_audio])

        # Propagate device choice only to runners that understand it (local ones).
        local_runner = m in {"vits", "melotts", "voxtream", "vibevoice", "chatterbox"}
        if local_runner:
            if args.cpu:
                cmd.append("--cpu")
            elif args.gpu:
                cmd.append("--gpu")
            else:
                # default: auto GPU if available (the runners also auto-detect)
                if common.gpu_is_available():
                    cmd.append("--gpu")

            if args.stream_local and m in {"melotts", "voxtream"}:
                cmd.append("--stream")

        if args.no_play:
            cmd.append("--no-play")
        if args.print_marks:
            cmd.append("--print-marks")

        logger.info("\n=== %s ===", m)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Runner failed for %s: %s", m, e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
