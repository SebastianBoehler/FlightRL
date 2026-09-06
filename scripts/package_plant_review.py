"""Package actual camera footage and a small local screenshot review."""

import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from flightrl.artifact_identity import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Packaging identifies its inputs; it cannot attest which code produced them.
    inputs = sorted(args.evaluation.glob("*.json")) + sorted(
        args.evaluation.glob("*.npz")
    )
    manifest = {
        "packager_sha256": sha256_file(__file__),
        "input_artifacts": {p.name: sha256_file(p) for p in inputs},
        "scope": "Derived camera review; original evaluation provenance unchanged",
    }
    (args.output / "package-manifest.json").write_text(json.dumps(manifest, indent=2))
    frames = np.load(args.evaluation / "learned-plant-sensor-recording.npz")["rgb"]
    h, w = frames.shape[1:3]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        "10",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output / "onboard-camera.mp4"),
    ]
    subprocess.run(command, input=frames.tobytes(), check=True)
    items = [
        ("01-plant-dark.png", "Three equipment rooms and the flown route"),
        ("02-plant-light.png", "Light theme"),
        ("03-camera-pose.png", "Recorded image placed at the drone camera pose"),
        ("04-link-loss.png", "Onboard recording continues through operator link loss"),
    ]
    cards = "".join(
        f'<figure><img src="{file}" alt="{title}"><figcaption>{title}</figcaption></figure>'
        for file, title in items
    )
    (args.output / "review.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Utility plant review</title>
<style>body{font:16px system-ui;background:#111820;color:#e6edf3;margin:40px auto;padding:0 24px;max-width:1200px}a{color:#83c8ee}h1{font-size:28px}p{line-height:1.6;color:#b9c6d2}figure{margin:32px 0}img{width:100%;border:1px solid #3a4856;border-radius:8px}figcaption{padding:12px 0}video{width:640px;max-width:100%;background:#000}</style>
<h1>Utility plant · visual review</h1><p><a href="/">Open the interactive workbench</a></p>
<p>Actual simulation recordings. The controller uses camera images and simulated depth. Materials, direct lighting, dust and the Metal lens pass affect the recorded images used for training. This remains a procedural simulation, not calibrated photorealism.</p>
<video controls loop src="onboard-camera.mp4"></video>"""
        + cards
    )
    print(args.output / "review.html")


if __name__ == "__main__":
    main()
