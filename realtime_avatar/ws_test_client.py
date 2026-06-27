"""Headless end-to-end test of the realtime avatar WebSocket (no browser).

Connects to the server, sends one line of text, collects the streamed audio +
frames, verifies they decode, saves samples, and prints latency metrics.

Run (after run_all.sh is up), from the main venv (has websockets via uvicorn[standard]):
    /home/pragay/WWAI/.venv/bin/python realtime_avatar/ws_test_client.py "Some text"
"""
import asyncio, base64, json, os, sys, time

import websockets  # provided by uvicorn[standard]

URL = os.environ.get("WS_URL", "ws://127.0.0.1:8000/ws")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_wstest_out")
os.makedirs(OUT, exist_ok=True)
TEXT = sys.argv[1] if len(sys.argv) > 1 else "Hello! This is a realtime test. The avatar should repeat this text."


async def main():
    t0 = time.time()
    first_audio = first_frame = None
    per_sentence = {}
    saved = 0
    async with websockets.connect(URL, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({"type": "speak", "text": TEXT}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=180)
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "start":
                print(f"start: {msg['n_sentences']} sentence(s), fps={msg['fps']}")
            elif t == "sentence":
                if first_audio is None:
                    first_audio = time.time() - t0
                audio = base64.b64decode(msg["audio_b64"])
                with open(os.path.join(OUT, f"sentence_{msg['idx']}.wav"), "wb") as f:
                    f.write(audio)
                per_sentence[msg["idx"]] = {"audio_bytes": len(audio), "frames": 0, "text": msg["text"]}
                print(f"  sentence[{msg['idx']}] audio={len(audio)}B  '{msg['text'][:50]}'")
            elif t == "frame":
                if first_frame is None:
                    first_frame = time.time() - t0
                per_sentence.setdefault(msg["s"], {"frames": 0})
                per_sentence[msg["s"]]["frames"] += 1
                if saved < 3:  # save first few frames to eyeball
                    jpg = base64.b64decode(msg["jpg_b64"])
                    with open(os.path.join(OUT, f"frame_s{msg['s']}_{msg['i']:04d}.jpg"), "wb") as f:
                        f.write(jpg)
                    saved += 1
            elif t == "sentence_end":
                print(f"  sentence[{msg['idx']}] end: {msg['n_frames']} frames")
            elif t == "error":
                print("ERROR:", msg["message"]); break
            elif t == "done":
                print("done.")
                break
    total = time.time() - t0
    nframes = sum(s.get("frames", 0) for s in per_sentence.values())
    print("-" * 50)
    print(f"time-to-first-audio: {first_audio*1000:.0f} ms" if first_audio else "no audio!")
    print(f"time-to-first-frame: {first_frame*1000:.0f} ms" if first_frame else "no frames!")
    print(f"total frames: {nframes}   wall: {total:.1f}s   saved samples in {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
