"""Fetch the licensed reference model at a fixed upstream revision, with hashes."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
REPOSITORY = "google-deepmind/mujoco_menagerie"
DESTINATION = Path(__file__).resolve().parents[1] / "assets/robots/xarm7"


def fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def main():
    tree = json.loads(
        fetch(
            f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
        )
    )
    entries = [
        e
        for e in tree["tree"]
        if e["type"] == "blob"
        and e["path"].startswith("ufactory_xarm7/")
        and ("/assets/" in e["path"] or e["path"].endswith((".xml", ".md", "LICENSE")))
    ]
    DESTINATION.mkdir(parents=True, exist_ok=False)

    def download(entry):
        relative = entry["path"].removeprefix("ufactory_xarm7/")
        data = fetch(
            f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{entry['path']}"
        )
        git_hash = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if git_hash != entry["sha"]:
            raise ValueError(f"Upstream blob mismatch: {relative}")
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return relative, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = dict(pool.map(download, entries))
    (DESTINATION / "manifest.json").write_text(
        json.dumps(
            dict(
                repository=REPOSITORY,
                revision=REVISION,
                license="BSD-3-Clause",
                files=files,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(files)} files in {DESTINATION}")


if __name__ == "__main__":
    main()
