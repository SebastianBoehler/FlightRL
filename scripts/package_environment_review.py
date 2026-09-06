"""Package recorded camera clips without rewriting their evaluation provenance."""

import argparse
import json
from pathlib import Path
import subprocess
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
    sections = []
    for name, title in [
        ("utility-plant", "Normal dust"),
        ("heavy-dust", "Heavy dust"),
        ("gusty-plant", "Stronger wind"),
    ]:
        with np.load(args.evaluation / f"{name}-sensor-recording.npz") as data:
            frames = data["rgb"][:200]
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "256x192",
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
                str(args.output / f"{name}.mp4"),
            ],
            input=frames.tobytes(),
            check=True,
        )
        sections.append(
            f'<section><h2>{title}</h2><video src="{name}.mp4" controls loop></video></section>'
        )
    (args.output / "review.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Environment engine review</title>
<style>body{font:16px system-ui;background:#121a22;color:#e9eef3;max-width:1280px;margin:32px auto;padding:0 24px}p{line-height:1.6;color:#b7c6d4}a{color:#83cbec}.clips{display:flex;gap:24px;flex-wrap:wrap}section{flex:1;min-width:256px}video{width:100%}img{width:100%;border:1px solid #364654;margin-top:24px}h1{font-size:28px}h2{font-size:18px}</style>
<h1>Coupled airflow and dust</h1><p><a href="/">Open interactive replay</a> · <a href="/data/evaluation.json">Recorded results</a></p><p>One frozen controller and one layout. Normal dust: 3/3 inspections. Heavy dust and stronger wind: 1/3 each. No collisions in these three runs. These clips show the first 20 seconds of actual camera recordings, with unchanged capture speed.</p><div class="clips">"""
        + "".join(sections)
        + """</div><img src="normal.png" alt="Normal dust replay"><img src="heavy.png" alt="Heavy dust replay"><p>Reduced-order airflow, rotor wakes and dust transport. Ideal depth. No retraining or broad generalization claim.</p>"""
    )
    print(args.output / "review.html")


if __name__ == "__main__":
    main()
