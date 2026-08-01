# Artifact lifecycle

`artifacts/` is a local cache, not source of truth. It is gitignored and a
checkpoint is never reproducible merely because its report names a Git commit.
Promotion requires a clean source commit, hashed dependencies/configuration,
held-out seeds, and the metric/model/runtime contracts used to produce it.

## Lifecycle values

- `active_candidate`: reproducible from clean `main` under the current metric
  and deployment contract.
- `historical_reference`: retained for regression or lineage but ineligible for
  promotion.
- `physical_evidence`: camera, telemetry, flash, or flight data that cannot be
  regenerated; cold-archive before removing a local copy.
- `regenerable_cache`: datasets, smoke runs, candidates, build outputs, and
  environments that can be deleted after their lineage is classified.

Authority is independent of lifecycle. `authority=none` means a file cannot be
used to justify shadow or flight control even when it is retained.

## 2026-08-01 baseline decision

Every pre-review learned navigation checkpoint used an incompatible mission,
policy, observation, action, or source-lineage contract. The learned stacks and
checkpoints were removed from the active workspace. No retained checkpoint
currently implements edge-v3 or has learned flight authority.

The July door-observability heads and their real-gate report were also archived:
the accepted head later failed the current native renderer, and the labeled real
set had become development data rather than an untouched promotion holdout. A
future edge-v3 student requires fresh frame-safe supervision, explicit dataset
hashes/splits, and preregistered real holdouts under the exact current contract.
The retained
[`aideck_frame_integrity_historical.json`](aideck_frame_integrity_historical.json)
registry establishes only coherent frame transport and is explicitly ineligible
for edge-v3 training, promotion, or flight authority.

`manifest.json` records an empty active-policy set and no portable retained
workspace evidence. Gitignored local firmware caches are deliberately excluded
because a fresh checkout cannot verify them. Raw AI Deck, Crazyflie, semantic,
hardware, and telemetry evidence must be preserved separately from regenerable
training outputs. New edge records must include parameter and quantized bytes,
MACs, peak activation and L1 use, actual GAP8 ELF L2 use, input bytes, firmware
identity, frame rate, latency, and sequence-parity results.

`local-archive://<date>/<path>` resolves below the current user's
`Documents/FlightRL-archive` directory. It is a same-disk convenience URI, not
a portable location or backup. The replay tree digest is reproducible from its
root with:

```bash
find . -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256
```

Historical firmware names and aggregate measurements are not active baselines.
Establish a fresh, portable identity and gate for any current binary.

## Cleanup rule

Copy irreplaceable evidence to the archive, verify byte count plus SHA-256 (or
a verified tree hash), and only then remove its working copy. Regenerable
outputs may be moved to Trash after the retained manifest is written. Never
bulk-delete `artifacts/` because it mixes caches with physical evidence.
The second first-review pass moved the remaining stale generated exports,
screenshots, evaluations, and parameter-count caches to the additional
recoverable Trash location recorded in the manifest.
