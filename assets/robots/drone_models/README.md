# Dimensioned drone references

These are Sunderlabs-authored reference meshes, **not manufacturer CAD or validated digital twins**. Rebuild deterministically with `.venv/bin/python scripts/build_drone_models.py`. Both native MuJoCo visuals and the browser consume these assets. No extra runtime dependencies.

| Reference | Overall dimensions (m) | Mass | Use |
|---|---|---|---|
| Avata 2 FPV | 0.185 × 0.212 × 0.064 | 0.377 kg | Live production and power scenes |
| Agras T25 | 2.585 × 2.675 × 0.780 | 32 kg | Live forest, battery installed and tank empty |

Size and mass: [DJI Avata 2 specifications](https://www.dji.com/avata-2/specs) and [DJI Agras T25 specifications](https://ag.dji.com/t25/specs), checked 2026-09-06. T25 also documents four 1.27 m propellers, a 1.925 m diagonal wheelbase, a 20 L spray tank, and two atomizers 1.368 m apart. Rotor centres are derived from its unfolded envelope and propeller diameter. The FPV is a guarded cinewhoop reference, not a five-inch racing frame.

Estimated details include internal layout, component shapes, FPV rotor layout, inertia, drag, thrust/rate limits and response constants. The T25's empty tank is visual; spraying, liquid slosh, payload changes and battery depletion are not simulated. Dimensions describe the full propeller envelope, represented conservatively as a rigid collision box. Rotating blades are acquisition-time visual animation, not measured motor RPM or simulated blade dynamics. MuJoCo retains fixed visual blade meshes internally; its exported browser sensor scene uses the timestamped animation, with the same collision proxy.

Camera mounts are explicit in the manifests. Both use the existing research RGB-D rig (63° vertical FOV, maximum depth 8 m), **not the manufacturer's camera optics or native depth sensor**. The FPV mount and dynamics are preserved to avoid silently changing the current controller's input contract. The agricultural controller uses its own mass and response profile; the existing frozen FPV actor cannot actuate it.

The forest wake uses each model's rotor positions and radii. Wake strength and ground return remain empirical approximations, not calibrated agricultural aerodynamics or CFD. Dust starts on surfaces below the aircraft rather than on its rotor collision proxy.

Each aircraft has ten draw meshes, grouped by material and rotor. Historical recorded scenarios keep their original geometry and physics metadata; these references apply to new live sessions.
