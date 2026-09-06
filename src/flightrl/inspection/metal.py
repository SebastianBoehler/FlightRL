"""Persistent Metal camera outputs owned by PyTorch's MPS allocator."""

from pathlib import Path
import torch


class MetalCamera:
    def __init__(self, scene, batch_size):
        if not torch.backends.mps.is_available() or not hasattr(
            torch.mps, "compile_shader"
        ):
            raise RuntimeError(
                "Metal camera requires PyTorch MPS compile_shader support"
            )
        self.batch_size = batch_size
        self.library = torch.mps.compile_shader(
            (Path(__file__).parents[1] / "native/inspection_camera.metal").read_text()
        )
        self.room = torch.tensor(
            scene.scenario.arrays["terrain_bounds"].copy(), device="mps"
        )
        self.boxes = torch.tensor(
            scene.scenario.arrays["terrain_obstacles"].copy(), device="mps"
        )
        self.panels = torch.tensor(scene.panels.copy(), device="mps")
        self.positions = torch.empty((batch_size, 3), device="mps")
        self.quaternions = torch.empty((batch_size, 4), device="mps")
        self.rgb = torch.empty((batch_size, 48, 64, 3), device="mps", dtype=torch.uint8)
        self.depth = torch.empty((batch_size, 48, 64), device="mps")

    def render(self, positions, quaternions):
        if positions.shape != (self.batch_size, 3) or quaternions.shape != (
            self.batch_size,
            4,
        ):
            raise ValueError("Metal camera pose batch mismatch")
        self.positions.copy_(torch.as_tensor(positions))
        self.quaternions.copy_(torch.as_tensor(quaternions))
        self.library.camera(
            self.positions,
            self.quaternions,
            self.room,
            self.boxes,
            self.panels,
            self.rgb,
            self.depth,
            len(self.boxes),
            len(self.panels),
            threads=[self.batch_size * 3072, 1, 1],
            group_size=[256, 1, 1],
        )
        return self.rgb, self.depth
