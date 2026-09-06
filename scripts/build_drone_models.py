"""Author dimensioned drone references, not manufacturer CAD or identified dynamics."""

import json
from pathlib import Path
from drone_meshes import Mesh
from flightrl.fleet.vehicles import VEHICLES

ROOT = Path(__file__).resolve().parents[1]


def build(kind):
    fpv = kind == "fpv"
    vehicle = VEHICLES[kind]
    radius = .0381 if fpv else .635
    xs, ys, z = (.05, .063, .016) if fpv else (.6575, .7025, .29)
    rotors = [[x, y, z] for x in (-xs, xs) for y in (-ys, ys)]
    carbon, shell, rubber, metal, glass, detail = [Mesh() for _ in range(6)]
    if fpv:
        shell.rounded_box([-.026, 0, .002], [.116, .056, .040], .010)
        shell.rounded_box([-.028, 0, .025], [.085, .05, .014], .006)
        rubber.rounded_box([-.028, 0, .014], [.054, .043, .024], .004)
        detail.box([-.041, 0, .027], [.012, .032, .004])
        shell.rounded_box([.030, 0, .009], [.018, .038, .033], .007)
        metal.tube([.035, -.021, .012], [.035, .021, .012], .008)
        glass.tube([.034, 0, .012], [.040, 0, .012], .011)
        for y in (-.027, .027):
            for x in (-.040, -.032, -.024, -.016):
                detail.box([x, y, .013], [.004, .0015, .012])
            rubber.box([-.035, y, -.029], [.042, .009, .006])
        for x, y, rz in rotors:
            carbon.tube([x * .4, y * .2, -.009], [x, y, -.009], .005)
            rubber.ring([x, y, .010], .040, .0025)
            rubber.ring([x, y, -.014], .040, .0025)
            for dx, dy in ((.040, 0), (-.040, 0), (0, .040), (0, -.040)):
                rubber.tube([x + dx, y + dy, -.014], [x + dx, y + dy, .010], .0015, 8)
            metal.tube([x, y, -.014], [x, y, .013], .009)
    else:
        shell.rounded_box([0, 0, .17], [.53, .43, .12], .045)
        shell.rounded_box([.045, 0, -.075], [.38, .37, .285], .055)
        rubber.rounded_box([-.055, 0, .31], [.29, .22, .16], .025)
        detail.box([-.055, 0, .388], [.16, .075, .004])
        metal.tube([-.18, 0, .20], [-.18, 0, .25], .06)  # Fill cap.
        shell.rounded_box([.276, 0, .15], [.05, .14, .08], .020)
        glass.tube([.298, 0, .15], [.305, 0, .15], .027)
        for y in (-.31, .31):
            for x in (-.24, .24):
                carbon.tube([x * .7, y * .5, .14], [x, y, -.365], .017)
            rubber.tube([-.43, y, -.371], [.43, y, -.371], .019)
        carbon.tube([-.30, -.684, -.16], [-.30, .684, -.16], .016)
        for y in (-.684, .684):
            metal.tube([-.30, y, -.16], [-.30, y, -.235], .026)
            detail.tube([-.30, y, -.24], [-.30, y, -.25], .052)
        for x, y, rz in rotors:
            carbon.tube([x * .23, y * .23, .18], [x, y, .20], .027)
            metal.tube([x * .47, y * .47, .18], [x * .53, y * .53, .18], .040)
            metal.tube([x, y, .175], [x, y, .265], .058)
            rubber.tube([x, y, .26], [x, y, .278], .062)
        for y in (-.19, .19):
            carbon.tube([-.24, y, .2], [-.27, y, .37], .008)
            for x in (-.12, -.08, -.04, 0, .04):
                detail.box([x, y * 1.14, .17], [.021, .003, .04])
    parts = [m.record(n, c, r, s) for m, n, c, r, s in (
        (carbon, "carbon", [.045, .050, .055], .7, .12),
        (shell, "shell", [.18, .20, .22] if fpv else [.65, .68, .64], .38, .05),
        (rubber, "rubber", [.08, .085, .09], .86, 0),
        (metal, "metal", [.28, .30, .32], .3, .75),
        (glass, "glass", [.018, .055, .075], .12, .45),
        (detail, "detail", [.21, .24, .22], .5, .2))]
    for i, position in enumerate(rotors):
        blade = Mesh(); blade.propeller(radius, .009 if fpv else .075, 3 if fpv else 2)
        blade.tube([0, 0, -.002], [0, 0, .003 if fpv else .025], .008 if fpv else .035)
        parts.append(blade.record(f"rotor_{i}", [.055, .060, .065], .55, .15, position))
    return dict(id=kind, name="Avata 2 FPV reference" if fpv else "Agras T25 agricultural reference",
                mass_kg=vehicle.mass_kg, dimensions_m=vehicle.dimensions_m,
                source=vehicle.source, rotor_radius_m=radius, rotor_centers_m=rotors,
                camera_offset_m=[.035, 0, .012] if fpv else [.305, 0, .15],
                provenance="Sunderlabs authored reference geometry; not manufacturer CAD",
                assumptions="Published overall size and mass; estimated component shapes and inertia. Research camera: 63 degree vertical FOV, 8 m depth. Response constants are uncalibrated.",
                payload="Battery installed; empty 20 L spray tank" if not fpv else "Battery installed",
                parts=parts)


if __name__ == "__main__":
    folder = ROOT / "assets/robots/drone_models"
    folder.mkdir(exist_ok=True)
    for kind in ("fpv", "agriculture"):
        (folder / f"{kind}.json").write_text(json.dumps(build(kind), separators=(",", ":")) + "\n")
