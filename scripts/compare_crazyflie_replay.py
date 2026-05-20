from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.replay import aligned_compare, compare, load_rows, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and compare Crazyflie-style real/sim replay CSV files")
    parser.add_argument("--real", required=True)
    parser.add_argument("--sim", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--align-time", action="store_true", help="resample sim signals onto real timestamps and report errors")
    parser.add_argument("--signals", nargs="*", default=None, help="optional signal list for --align-time")
    args = parser.parse_args()

    real_rows = load_rows(args.real)
    result = {"real": summarize(real_rows)}
    if args.sim:
        sim_rows = load_rows(args.sim)
        result["sim"] = summarize(sim_rows)
        result["delta"] = compare(result["real"], result["sim"])
        if args.align_time:
            result["aligned"] = aligned_compare(real_rows, sim_rows, args.signals)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"wrote {output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
