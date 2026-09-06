"""Immutable, serializable environment settings; metres, seconds and world FLU."""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    surface_style: str = "industrial"
    wind_m_s: tuple[float, float, float] = (0.12, 0, 0)
    turbulence_m_s: float = 0.12
    correlation_s: float = 0.6
    air_drag_per_s: float = 0.35
    dust_extinction_per_m: float = 0.035
    grain_diameter_um: tuple[float, float] = (10.0, 40.0)
    grain_density_kg_m3: float = 2500.0
    particle_count: int = 512
    settled_fraction: float = 0.0
    dust_bed_bounds: tuple[float, float, float, float] | None = None
    resuspension_m_s: float = 0.35
    downwash_m_s: float = 1.5
    ambient: float = 0.42
    wall_rgb: tuple[float, float, float] = (159, 169, 167)
    equipment_rgb: tuple[float, float, float] = (49, 83, 96)
    floor_rgb: tuple[float, float, float] = (87, 98, 103)
    equipment_roughness: float = 0.35
    equipment_metalness: float = 0.65
    floor_roughness: float = 0.65
    floor_metalness: float = 0.12
    wall_roughness: float = 0.8
    wall_metalness: float = 0.03
    sun_direction: tuple[float, float, float] = (0.25, 0.65, 0.72)
    sun_strength: float = 0
    lights: tuple[
        tuple[float, ...], ...
    ] = ()  # xyz, linear RGB tint, relative intensity
    windows: tuple[tuple[float, ...], ...] = ()  # glass AABBs on room boundary

    def __post_init__(self):
        if self.surface_style not in ("industrial", "data_center", "forest"):
            raise ValueError("surface_style")
        diameter = np.asarray(self.grain_diameter_um)
        if (
            diameter.shape != (2,)
            or not np.isfinite(diameter).all()
            or not 1 <= diameter[0] <= diameter[1] <= 100
        ):
            raise ValueError("grain_diameter_um must be within 1..100 micrometres")
        if self.grain_density_kg_m3 <= 1.225:
            raise ValueError("dust density must exceed air density")
        object.__setattr__(self, "grain_diameter_um", tuple(float(x) for x in diameter))
        if self.dust_bed_bounds is not None:
            bed = np.asarray(self.dust_bed_bounds)
            if (
                bed.shape != (4,)
                or not np.isfinite(bed).all()
                or np.any(bed[::2] >= bed[1::2])
            ):
                raise ValueError("dust_bed_bounds")
            object.__setattr__(self, "dust_bed_bounds", tuple(float(x) for x in bed))
        if not self.name:
            raise ValueError("environment name required")
        for name in (
            "wind_m_s",
            "wall_rgb",
            "equipment_rgb",
            "floor_rgb",
            "sun_direction",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(name)
            object.__setattr__(self, name, tuple(float(x) for x in value))
            if name.endswith("rgb") and ((value < 0) | (value > 255)).any():
                raise ValueError(name)
        for name in (
            "turbulence_m_s",
            "air_drag_per_s",
            "dust_extinction_per_m",
            "grain_density_kg_m3",
            "downwash_m_s",
            "ambient",
            "sun_strength",
            "resuspension_m_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name)
        for name in ("correlation_s",):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name)
        for name in (
            "settled_fraction",
            "equipment_roughness",
            "equipment_metalness",
            "floor_roughness",
            "floor_metalness",
            "wall_roughness",
            "wall_metalness",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(name)
        if type(self.particle_count) is not int or not 1 <= self.particle_count <= 8192:
            raise ValueError("particle_count")
        if np.linalg.norm(self.sun_direction) < 1e-6:
            raise ValueError("sun_direction")
        object.__setattr__(self, "lights", tuple(tuple(row) for row in self.lights))
        object.__setattr__(self, "windows", tuple(tuple(row) for row in self.windows))
        for rows, size in ((self.lights, 7), (self.windows, 6)):
            if len(rows) > 64:
                raise ValueError("too many lights/windows")
            if any(len(row) != size or not np.isfinite(row).all() for row in rows):
                raise ValueError("light/window row")
        if any(min(row[3:]) < 0 for row in self.lights):
            raise ValueError("negative light")
        if any(any(row[i] >= row[i + 1] for i in (0, 2, 4)) for row in self.windows):
            raise ValueError("window bounds")

    def render_buffers(self):
        direction = np.array(self.sun_direction) / np.linalg.norm(self.sun_direction)
        settings = np.array(
            [
                self.ambient,
                *self.wall_rgb,
                *self.equipment_rgb,
                self.equipment_roughness,
                self.equipment_metalness,
                *direction,
                self.sun_strength,
                *self.floor_rgb,
                self.floor_roughness,
                self.floor_metalness,
                self.wall_roughness,
                self.wall_metalness,
                ("industrial", "data_center", "forest").index(self.surface_style),
            ],
            np.float32,
        )
        return (
            settings,
            np.array(self.lights, np.float32).reshape(-1, 7),
            np.array(self.windows, np.float32).reshape(-1, 6),
        )

    def report(self):
        return asdict(self)
