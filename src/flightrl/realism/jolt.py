"""Small typed bridge to the pinned Jolt build; no alternate collision backend."""

import ctypes as C
import sys
from pathlib import Path
import numpy as np

F = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
U = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")


class NativeWorld:
    def __init__(self):
        path = (
            Path(__file__).resolve().parents[3]
            / "build/realism"
            / (
                "libflightrl_contacts.dylib"
                if sys.platform == "darwin"
                else "libflightrl_contacts.so"
            )
        )
        if not path.is_file():
            raise RuntimeError(
                "Build native contacts first: python scripts/build_realism_physics.py"
            )
        self.lib = C.CDLL(str(path))
        signatures = {
            "create": ([], C.c_void_p),
            "destroy": ([C.c_void_p], None),
            "error": ([], C.c_char_p),
            "mesh": ([C.c_void_p, F, C.c_int, U, C.c_int], C.c_int),
            "box": ([C.c_void_p, F, F, F, C.c_float], C.c_int),
            "step": ([C.c_void_p, C.c_float], C.c_int),
            "state": ([C.c_void_p, C.c_int, F], None),
            "force": ([C.c_void_p, C.c_int, F], None),
            "angular": ([C.c_void_p, C.c_int, F], None),
            "velocity": ([C.c_void_p, C.c_int, F, F], None),
            "transform": ([C.c_void_p, C.c_int, F, F], None),
            "contacts": ([C.c_void_p, F, C.c_int], C.c_int),
            "rays": ([C.c_void_p, F, F, C.c_int, C.c_float, F], None),
        }
        for name, (args, result) in signatures.items():
            f = getattr(self.lib, "fr_" + name)
            f.argtypes = args
            f.restype = result
        self.ptr = self.lib.fr_create()
        self._count = 0

    def close(self):
        if self.ptr:
            self.lib.fr_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        if getattr(self, "ptr", None):
            self.close()

    def _created(self, value):
        if value < 0:
            raise RuntimeError(
                self.lib.fr_error().decode() or "Jolt body allocation failed"
            )
        self._count += 1
        return value

    def mesh(self, vertices, indices):
        return self._created(
            self.lib.fr_mesh(
                self.ptr,
                np.ascontiguousarray(vertices, np.float32),
                len(vertices),
                np.ascontiguousarray(indices, np.uint32),
                len(indices),
            )
        )

    def box(self, spec):
        return self._created(
            self.lib.fr_box(
                self.ptr,
                *[
                    np.array(spec[k], np.float32)
                    for k in ("position", "quaternion", "halfExtents")
                ],
                float(spec["mass"]),
            )
        )

    def _id(self, handle):
        if self.ptr is None or not 0 <= handle < self._count:
            raise ValueError("Invalid Jolt body handle")
        return handle

    def state(self, handle):
        out = np.empty(13, np.float32)
        self.lib.fr_state(self.ptr, self._id(handle), out)
        return out

    def get_body_stats(self, handle):
        a = self.state(handle)
        return a[:3], a[3:7], a[7:10]

    def get_velocity(self, handle):
        return self.state(handle)[7:10]

    def get_angular_velocity(self, handle):
        return self.state(handle)[10:13]

    def step(self, dt):
        if dt == 0:
            return
        if not np.isfinite(dt) or not 0 < dt <= 0.02:
            raise ValueError("Jolt step must be in (0,.02]")
        error = self.lib.fr_step(self.ptr, float(dt))
        if error:
            raise RuntimeError(f"Jolt update failed, capacity flags={error}")

    def apply_force(self, handle, *force):
        self.lib.fr_force(self.ptr, self._id(handle), np.array(force, np.float32))

    def set_angular_velocity(self, handle, *omega):
        self.lib.fr_angular(self.ptr, self._id(handle), np.array(omega, np.float32))

    def set_linear_velocity(self, handle, *velocity):
        self.lib.fr_velocity(
            self.ptr,
            self._id(handle),
            np.array(velocity, np.float32),
            self.get_angular_velocity(handle).copy(),
        )

    def set_transform(self, handle, position, quaternion):
        self.lib.fr_transform(
            self.ptr,
            self._id(handle),
            np.array(position, np.float32),
            np.array(quaternion, np.float32),
        )

    def contacts(self):
        out = np.empty((256, 4), np.float32)
        count = self.lib.fr_contacts(self.ptr, out, len(out))
        return out[:count]

    def rays(self, starts, directions, distance):
        starts = np.asarray(starts, np.float32)
        directions = np.asarray(directions, np.float32)
        if starts.ndim != 2 or starts.shape[1] != 3 or directions.shape != starts.shape:
            raise ValueError(
                "Ray origins and directions must have matching N x 3 shapes"
            )
        if (
            not np.isfinite(starts).all()
            or not np.isfinite(directions).all()
            or not np.isfinite(distance)
            or distance <= 0
        ):
            raise ValueError("Rays must be finite with positive distance")
        out = np.empty((len(starts), 5), np.float32)
        self.lib.fr_rays(
            self.ptr,
            np.ascontiguousarray(starts, np.float32),
            np.ascontiguousarray(directions, np.float32),
            len(starts),
            float(distance),
            out,
        )
        return {
            "fraction": out[:, 0],
            "normal": out[:, 1:4],
            "body": out[:, 4].astype(int),
        }
