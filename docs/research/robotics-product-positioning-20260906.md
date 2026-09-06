# Robotics product focus after the workbench review

2026-09-06 · Objective: industrial pilots and hardware partners.

The strongest initial customer is an industrial robotics integrator or OEM
engineering team trying to validate an inspection or production-cell task on its
own robot. Sell a bounded integration and validation pilot: one model, one task,
a repeatable failure case, and recorded evidence of the resulting behavior.

## What could differentiate this product

A shared loop from model and actuator import, through physics and sensor errors,
to controller decisions and replayable outcomes. Measure time to integrate the
partner's model, time to explain a failed decision, and held-out task performance.
These are hypotheses to test with partners, not established product-market fit.

Synchronization and visualization alone are not a new category. Rerun already
supports synchronized images, spatial data and time series, including robotics
workflows. Foxglove provides robotics visualization and time-based plots. They
are credible comparison tools and potential integration surfaces, rather than
features we should assume nobody has implemented.

- [Rerun official repository](https://github.com/rerun-io/rerun)
- [Rerun overview](https://rerun.io/docs/overview/what-is-rerun)
- [Foxglove plots](https://docs.foxglove.dev/docs/visualization/panels/plot)

The exact University of Tübingen LinkedIn post could not be identified in the
bounded search. Do not attribute a product or novelty claim to that university
without the post. Related autonomous-driving research is not proof of a match.

## China demo and the next evidence milestone

Use production inspection to introduce the drone/rover workflow, then show the
imported xArm7 and replay a camera capture alongside its actual joint state.
Explain which controllers are learned and which are reference joint servos.
Ask prospective partners for a representative robot model, actuator interface,
sensor calibration and one costly failure scenario. A signed pilot and access
to those inputs matter more than adding several unrelated robot categories.

Next, add a bounded arm task with success/contact criteria and a repeatable
reference-controller baseline; train only after that environment is validated.
Compare debugging a held-out failure with the partner's existing toolchain.
Real sensor ingestion then needs explicit clock-domain mapping, offset/drift and
uncertainty measurements, calibration provenance and hardware validation. Our
single simulation clock does not establish trustworthy multi-device clock sync.

Keep agricultural airframe identification, humanoid locomotion and marine
physics dependent on a concrete partner use case. A mesh alone is insufficient
for an accurate physical model or reusable policy.
