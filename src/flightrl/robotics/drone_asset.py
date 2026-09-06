"""Authored, dimensioned reference assets shared by native and browser scenes."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
from flightrl.fleet.vehicles import VEHICLES

ASSETS = Path(__file__).resolve().parents[3] / "assets/robots/drone_models"


@lru_cache
def drone_model(kind):
    if kind not in ("fpv", "agriculture"):
        raise ValueError(f"Unsupported drone reference: {kind}")
    model = json.loads((ASSETS / f"{kind}.json").read_text())
    vehicle = VEHICLES[kind]
    if model["mass_kg"] != vehicle.mass_kg or tuple(model["dimensions_m"]) != vehicle.dimensions_m:
        raise ValueError("Drone reference disagrees with the physical vehicle catalog")
    return model


def model_identity(kind):
    model = {k: v for k, v in drone_model(kind).items() if k != "parts"}
    model["asset_sha256"] = hashlib.sha256((ASSETS / f"{kind}.json").read_bytes()).hexdigest()
    return model


def attach_fpv(xml):
    model = drone_model("fpv")
    body = xml.find("worldbody/body[@name='drone']")
    # Existing mass and bounding-box inertia are deliberate simulation proxies.
    body.find("geom[@name='drone_chassis']").set("group", "3")
    asset = ET.SubElement(xml, "asset")
    for part in model["parts"]:
        name = f"drone_visual_{part['name']}"
        ET.SubElement(asset, "mesh", name=name,
                      vertex=" ".join(map(str, part["vertices"])),
                      face=" ".join(map(str, part["indices"])))
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name,
                      pos=" ".join(map(str, part["position"])),
                      rgba=" ".join(map(str, part["color"] + [1])),
                      mass="0", contype="0", conaffinity="0", group="2")


def fpv_source_identity(source):
    return hashlib.sha256(source.read_bytes() + (ASSETS / "fpv.json").read_bytes()).hexdigest()
