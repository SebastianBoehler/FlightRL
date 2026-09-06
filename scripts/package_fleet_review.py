"""Package the saved fleet pilot and verify its reconstructed scene identity."""
import hashlib
import json
from pathlib import Path

import numpy as np
from flightrl.inspection.environments import environment_scene
from flightrl.fleet.rollout import free_points
from flightrl.fleet.vehicles import VEHICLES

root = Path(__file__).resolve().parents[1]
source = root / 'artifacts/fleet-pilot-20260905-v2/replay.json'
data = json.loads(source.read_text())
scene = environment_scene('forest', 120)
a = scene.scenario.arrays
sites = free_points(a['terrain_bounds'], a['terrain_obstacles'], VEHICLES['fpv'].radius, np.random.default_rng(120), 6)
if not np.allclose(sites[:3], data['records'][0]['positions']):
    raise ValueError('Reconstructed scene/spawn does not match the saved pilot')
if not np.allclose(sites[3:], data['records'][0]['goals']):
    raise ValueError('Reconstructed assignments do not match the saved pilot')
data['scene'] = {'boxes': a['terrain_obstacles'].tolist(), 'room': a['terrain_bounds'].tolist()}
data['provenance'] = {
    'family': 'forest', 'seed': 120,
    'replay_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'camera': 'Detailed re-render at recorded poses; not training observations',
    'communication': '5 Hz peer messages · 200 ms delay · 1 s expiry; no shared images',
    'vehicle': 'FPV · Avata 2 size reference',
    'dimensions': list(VEHICLES['fpv'].dimensions_m),
}
target = root / 'viewer/public/fleet/pilot.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data))
print(target)
