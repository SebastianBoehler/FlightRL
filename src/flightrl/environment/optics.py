"""Metal sensor-only lens response. Simulation observer never receives this pass."""

from pathlib import Path
import torch


class CameraOptics:
    def __init__(self, width=256, height=192):
        if not torch.backends.mps.is_available() or not hasattr(
            torch.mps, "compile_shader"
        ):
            raise RuntimeError(
                "Enhanced plant camera requires local PyTorch MPS compile_shader"
            )
        shader = Path(__file__).parents[1] / "native/inspection_optics.metal"
        self.library = torch.mps.compile_shader(shader.read_text())
        self.width, self.height = width, height
        self.source = torch.empty((height, width, 3), dtype=torch.uint8, device="mps")
        self.output = torch.empty_like(self.source)
        self.frame = 0

    def apply(self, rgb):
        self.source.copy_(torch.from_numpy(rgb))
        self.library.optics(
            self.source,
            self.output,
            self.width,
            self.height,
            self.frame,
            threads=[self.height * self.width, 1, 1],
            group_size=[256, 1, 1],
        )
        rgb[:] = self.output.cpu().numpy()
        self.frame += 1
