# Rotor dust and diagnostic view

The utility plant uses 4,096 representative dust parcels, with 80% initially settled. The rotor demonstration raises the settled fraction to 98% and reduces ambient turbulence to isolate the wake. Four downwash jets and a reduced-order ground return flow entrain the finite bed; no particles are created. Settling and swept obstacle contacts remain active. This is an illustrative flow approximation, not calibrated CFD.

The 3D observer displays enlarged amber particle glyphs, cyan airflow arrows (0.5 display metres per m/s, capped at 0.8 m), and orange wind-induced acceleration (2 display metres per m/s², capped at 1.5 m). These are diagnostic overlays, excluded from the sensor. RGB uses the same airborne population for shadowed volume extinction/scattering and depth-tested soft parcel projections. Native depth stays ideal.

`PYTHONPATH=src .venv/bin/python scripts/record_rotor_dust.py` records a 30-second frozen-policy episode to `artifacts/rotor-dust-20260905b` (requires that destination not exist). The current capture has 669 resuspension events, no collision, and one of three panels validated before its 30-second budget ends. It is a visual demonstration, not a complete navigation benchmark. Earlier episodes remain selectable and retain their original physics.

Verification: 44 focused environment, sensor, autonomy, replay and source-packaging tests passed before the diagnostic sampling refinement; 21 environment/industrial tests rerun afterward. Viewer production build passed; browser reports no errors or warnings. The bed test checks zero-thrust inactivity, thrust-driven lift and finite parcel count.

## Physics audit and corner experiment

The first rotor replay revealed two artifacts: a discontinuous uplift cutoff at drone height formed a horizontal particle sheet, and floor clipping at +1 mm prevented slow particles from reaching the deposition condition at zero. Both are corrected. Ground-return velocity now decays smoothly with height; floor contact uses the same numerical surface as clipping.

Particle response and terminal velocity are no longer independent artistic parameters. Grain diameter and material density determine Stokes response time; gravity includes buoyancy and drag uses the Schiller–Naumann Reynolds correction. These are spherical-grain approximations in room-temperature air. Wake/entrainment still use a reduced-order prescribed field and empirical threshold; this is not a Navier–Stokes solve or validated rotor aerodynamics. Optical extinction remains an authored coefficient, not measured dust mass calibration.

Reference: [NIST sedimentation guidance](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication960-1.pdf) and [OpenFOAM Schiller–Naumann correlation](https://cpp.openfoam.org/v12/SchillerNaumann_8C_source.html).

`record_corner_dust.py` creates a separate scripted 60-second native-flight experiment with 8,192 representative grains of 20–60 μm, all initially on a localized floor patch. It approaches, stirs, climbs, retreats, and observes settling. The viewer uses small 7 mm diagnostic glyph radii to make representative parcels visible; those radii are not physical grain sizes and never enter the sensor. The camera uses spatial extinction and scattering of the actual airborne population.

47 focused tests pass after the audit, including still-air terminal settling, local-bed persistence without flow, deposition, mass conservation, and wake continuity across drone height. Earlier immutable recordings show their older models and should not be used as evidence for this correction.

## Room coordinates

The current `corner-dust-20260905-world` replay records settled and airborne positions in world coordinates. The observer shows both populations, including the initial floor deposit. Airflow diagnostic samples are now fixed to the room rather than translated with the drone. A regression moves the drone between distant poses with rotors off and verifies that neither dust positions nor airflow sample coordinates move. Only sampled velocities change when a wake is present. The prescribed wake responds immediately to rotor pose; advected fluid momentum and wake history remain outside this simplified model.

## Wall contact and rotor-plane correction

A follow-up found that vertical-wall hits incorrectly marked grains as deposited,
freezing them above the floor. Wall contact now removes inward normal velocity and
retains tangential falling motion; deposition requires supporting floor/top contact.
The wake now decays smoothly above the rotor plane, and the empirical ground-return
term tapers below it instead of creating an opposing-flow discontinuity.

21 focused environment/contact tests pass. The 60-second corner replay was regenerated
in artifacts/corner-dust-20260905-contact-fix and the active viewer links updated.
Motor power was unchanged. Absolute wake speed remains a prescribed reduced-order
parameter, not a rotor-disk/thrust calibration. This does not establish CFD accuracy.
