# Research notes

Only current contracts and literature notes belong in this directory. Historical
run logs, continuation prompts, and superseded live instructions were removed
from the active tree after the first repository review because they described
invalidated checkpoints and polluted experiment context.

The full historical text remains recoverable from Git commit `038ee15`. Its
artifacts have no flight authority; the retained catalog is
`docs/evidence/manifest.json`.

Current starting points:

- `../robotics-workbench-20260906.md` — implemented xArm7 import, causal recording, synchronized replay and measured local validation.
- `robotics-product-positioning-20260906.md` — industrial pilot focus, existing visualization tools and the next partner evidence milestone.
- `robotics-retrospective-20260906.md` — current import/timestamp gaps, articulated-model choice, unified workbench and measured performance gates.
- `../robotics-inspection-20260906.md` — implemented shared drone/rover physics, actual RGB-D training, and held-out inspection/docking evidence.
- `robotics-platform-direction-20260906.md` — staged general robotics capability map and product direction.

- `realism-implementation-20260906.md` — shared forest RGB-D, Jolt contacts and measured local rendering budget.

- `autonomous_drone_technology_trajectory_20260903.md` — current technology
  trajectory, durable architecture bets, and explicit kill criteria.
- `cross_airframe_autonomy_stack_20260902.md` — primary-source-backed language,
  accelerator, bundle, deployment, and cross-airframe architecture decision.
- `docs/edge_navigation_v3.md` — edge-shaped policy and deployment boundary.
- `docs/evidence/README.md` — artifact lifecycle and authority rules.
- `architecture_literature_review.md` — literature comparison.
- `pulp_dronet_2019_analysis.md` — embedded perception reference.
- `vision_observation_contract.md` — visual-input rationale and the
  edge-v3 supersession boundary.

New experiment notes should bind claims to a clean commit, exact metric and
model contracts, disjoint evaluation seeds, and hashed outputs. A simulation or
teacher result never grants hardware authority.
