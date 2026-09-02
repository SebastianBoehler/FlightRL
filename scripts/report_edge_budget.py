from __future__ import annotations

import argparse
import json

from flightrl.puffer4_edge_budget import edge_actor_budget
from flightrl.puffer4_edge_policy import EdgeNavigationActor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the executable edge-v3 actor's static deployment budget",
    )
    parser.add_argument("--hidden-size", type=int, default=48)
    args = parser.parse_args()

    budget = edge_actor_budget(EdgeNavigationActor(hidden_size=args.hidden_size))
    print(json.dumps(budget, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
