from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flightrl.puffer4_door_policy import (
    DOOR_HEIGHT,
    DOOR_PIXELS,
    DOOR_POLICY_OBS_DIM,
    DOOR_WIDTH,
)
from scripts.train_puffer_fixed_door import DEFAULT_PUFFER, load_puffer


def main() -> None:
    args = parse_args()
    config, torch_pufferl = load_puffer(args.puffer_root, args.env_name)
    config["env"]["seed"] = args.seed
    config["env"]["appearance_seed"] = args.appearance_seed
    config["env"]["layout_diversity"] = float(args.expanded_layouts)
    config["env"]["camera_randomization"] = float(
        args.hardware_camera_randomization
    )
    config["env"]["obstacle_probability"] = args.obstacle_probability
    config["vec"]["total_agents"] = max(args.samples, 1)
    vec = torch_pufferl._C.create_vec(config, torch_pufferl._C.gpu)
    observations = torch_pufferl._cpu_tensor(
        vec.obs_ptr,
        (vec.total_agents, vec.obs_size),
        torch.float32,
    )
    vec.reset()
    frames = observations[: args.samples, :DOOR_PIXELS].reshape(
        -1,
        DOOR_HEIGHT,
        DOOR_WIDTH,
    )
    labels = observations[
        : args.samples,
        DOOR_POLICY_OBS_DIM + 2 : DOOR_POLICY_OBS_DIM + 6,
    ]
    sheet = contact_sheet(frames, labels, columns=args.columns, scale=args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    vec.close()
    print(args.output.resolve())


def contact_sheet(
    frames: torch.Tensor,
    labels: torch.Tensor,
    *,
    columns: int,
    scale: int,
) -> Image.Image:
    rows = (frames.shape[0] + columns - 1) // columns
    tile_width = DOOR_WIDTH * scale
    tile_height = DOOR_HEIGHT * scale
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height))
    draw = ImageDraw.Draw(sheet)
    for index, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        image = Image.fromarray(
            (frame.clamp(0.0, 1.0).numpy() * 255.0).astype("uint8"),
            mode="L",
        ).resize((tile_width, tile_height), Image.Resampling.NEAREST)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(image.convert("RGB"), (x, y))
        visible = float(label[0]) > 0.5
        color = (40, 220, 90) if visible else (230, 70, 70)
        draw.rectangle(
            (x, y, x + tile_width - 1, y + tile_height - 1),
            outline=color,
            width=max(1, scale),
        )
        if visible:
            center_x = x + float(label[1]) * tile_width
            center_y = y + float(label[2]) * tile_height
            radius = max(2, 2 * scale)
            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                outline=color,
                width=max(1, scale),
            )
    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render exact fixed-door Puffer observations as a contact sheet."
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER)
    parser.add_argument("--env-name", default="flightrl_fixed_door_d1")
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10_011)
    parser.add_argument("--appearance-seed", type=int, default=10_007)
    parser.add_argument("--obstacle-probability", type=float, default=0.5)
    parser.add_argument("--expanded-layouts", action="store_true")
    parser.add_argument(
        "--hardware-camera-randomization",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/puffer_fixed_door_composed/contact_sheet.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
