"""Pinned upstream learned RGB-only model on a declared development sequence.

Run from repository root with the upstream checkout on PYTHONPATH. No ground
truth, depth or calibrated poses enter inference. Evaluation is a later stage.
"""

import hashlib
import json
import inspect
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from flightrl.reconstruction.experiment import experiment
from flightrl.reconstruction.mps import prepare_rotary_cache
from flightrl.reconstruction.geometry import axial_depth_points
from flightrl.fleet.camera_policy.network import Policy

upstream = Path(inspect.getfile(GCTStream)).resolve().parents[2]
expected_commit = "bfcd0f20383d3a35cc9757a36ab1d5b6e5064df4"
actual_commit = subprocess.check_output(
    ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
).strip()
if (
    actual_commit != expected_commit
    or subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    ).strip()
):
    raise ValueError("LingBot checkout must match the clean pinned upstream commit")
checkpoint_revision = "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9"
checkpoint = hf_hub_download(
    "robbyant/lingbot-map", "lingbot-map.pt", revision=checkpoint_revision
)
with open(checkpoint, "rb") as stream:
    checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
if (
    checkpoint_sha256
    != "ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72"
):
    raise ValueError("LingBot checkpoint digest does not match the published trial")

folder = Path("artifacts/reconstruction-20260906/learned-trial")
folder.mkdir(exist_ok=False)
plan = dict(
    model="robbyant/lingbot-map",
    upstream_commit=actual_commit,
    checkpoint_revision=checkpoint_revision,
    seed=4000,
    drone=1,
    frame_stride=10,
    num_scale_frames=4,
    input="RGB only",
    device="mps",
    dtype="float32",
    attention="SDPA",
    image_size=[266, 196],
    scope="Development smoke evaluation, not held-out generalization or online control",
)
(folder / "plan.json").write_text(json.dumps(plan, indent=2))
print("Generate development RGB sequence", flush=True)
_, review, frames = experiment(
    4000, Policy("artifacts/camera-control-linkloss-20260906/actor.pt")
)
indices = np.arange(0, len(frames), 10)
input_rgb = np.stack([frames[i][:, :256] for i in indices])
np.save(folder / "input-rgb.npy", input_rgb)
(folder / "reference.json").write_text(json.dumps(review))
images = (
    torch.from_numpy(
        np.stack(
            [
                np.asarray(
                    Image.fromarray(im).resize((266, 196), Image.Resampling.BICUBIC)
                ).copy()
                for im in input_rgb
            ]
        )
    )
    .permute(0, 3, 1, 2)
    .float()
    / 255
)
print("Build and load model", flush=True)
model = GCTStream(
    img_size=518,
    patch_size=14,
    enable_3d_rope=True,
    use_sdpa=True,
    kv_cache_sliding_window=16,
    kv_cache_scale_frames=4,
)
state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
model.load_state_dict(state.get("model", state), strict=True)
del state
print("Prepared rotary caches:", prepare_rotary_cache(model), flush=True)
torch.mps.set_per_process_memory_fraction(0.65)
model = model.to("mps").eval()
torch.mps.synchronize()
start = time.perf_counter()
with torch.inference_mode():
    predictions = model.inference_streaming(
        images,
        num_scale_frames=4,
        keyframe_interval=1,
        output_device=torch.device("cpu"),
    )
torch.mps.synchronize()
elapsed = time.perf_counter() - start
extrinsics, intrinsics = pose_encoding_to_extri_intri(
    predictions["pose_enc"], images.shape[-2:]
)
poses = np.tile(np.eye(4), (len(indices), 1, 1))
poses[:, :3, :4] = extrinsics[0].numpy()
# Published benchmark/methods/lingbot_map.py: decoded output is C2W.
# Do not invert it according to the inherited utility docstring.
points = np.stack(
    [
        axial_depth_points(d[..., 0], k, pose)
        for d, k, pose in zip(
            predictions["depth"][0].numpy(), intrinsics[0].numpy(), poses
        )
    ]
)
np.savez_compressed(
    folder / "predictions.npz",
    poses=poses,
    points=points,
    depth=predictions["depth"][0].numpy(),
    confidence=predictions["depth_conf"][0].numpy(),
    indices=indices,
    colors=images.permute(0, 2, 3, 1).numpy(),
    intrinsics=intrinsics[0].numpy(),
)
result = dict(
    **plan,
    frames=len(indices),
    inference_s=elapsed,
    frames_per_s=len(indices) / elapsed,
    checkpoint_sha256=checkpoint_sha256,
)
(folder / "runtime.json").write_text(json.dumps(result, indent=2))
print(result, flush=True)
