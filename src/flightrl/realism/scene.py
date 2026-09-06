"""Validate the exact solid geometry exported by the detailed scene builder."""

import base64
import hashlib
import json
import numpy as np
from flightrl.robotics.drone_asset import drone_model


def decode_scene(payload):
    if (
        payload.get("schema"),
        payload.get("units"),
        payload.get("up"),
        payload.get("quaternionOrder"),
    ) != ("flightrl.shared_forest.v1", "m", "z", "xyzw"):
        raise ValueError("Shared scene must declare metres, Z-up and xyzw rotations")
    vertices = np.frombuffer(
        base64.b64decode(payload["vertices"], validate=True), "<f4"
    ).reshape(-1, 3)
    indices = np.frombuffer(
        base64.b64decode(payload["indices"], validate=True), "<u4"
    ).reshape(-1, 3)
    if not 1 <= len(indices) <= 500_000 or len(indices) != payload["triangleCount"]:
        raise ValueError("Invalid shared triangle count")
    if (
        not np.isfinite(vertices).all()
        or np.abs(vertices).max() > 2048
        or indices.max() >= len(vertices)
    ):
        raise ValueError("Invalid shared geometry coordinates or indices")
    a, b, c = vertices[indices].transpose(1, 0, 2)
    if (np.linalg.norm(np.cross(b - a, c - a), axis=1) < 1e-8).any():
        raise ValueError("Degenerate solid triangle")
    wind = np.asarray(payload["wind_m_s"], float)
    if wind.shape != (3,) or not np.isfinite(wind).all():
        raise ValueError("Finite world wind vector required")
    bodies = payload["bodies"]
    if not 3 <= len(bodies) <= 32 or len({x["id"] for x in bodies}) != len(bodies):
        raise ValueError("Unique body IDs and three drones required")
    for body in bodies[:3]:
        reference = drone_model(body.get("vehicle"))
        if not np.isclose(body["mass"], reference["mass_kg"]):
            raise ValueError("Body mass must match the selected unloaded drone reference")
        if not np.allclose(np.asarray(body["halfExtents"]) * 2, reference["dimensions_m"]):
            raise ValueError("Collision envelope must match the selected drone reference")
        if "model" in body and body["model"] != {k: v for k, v in reference.items() if k != "parts"}:
            raise ValueError("Displayed drone reference differs from the native asset")
    for body in bodies:
        p, q, size = (
            np.asarray(body[k], float)
            for k in ("position", "quaternion", "halfExtents")
        )
        if p.shape != (3,) or q.shape != (4,) or size.shape != (3,):
            raise ValueError("Invalid rigid body dimensions")
        if not np.isfinite(np.r_[p, q, size, body["mass"]]).all() or not np.isclose(
            np.linalg.norm(q), 1
        ):
            raise ValueError("Finite body values and unit quaternions required")
        if (size <= 0).any() or (size > 2).any() or not 0 < body["mass"] <= 100:
            raise ValueError("Invalid rigid body mass or extents")
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return vertices, indices, digest
