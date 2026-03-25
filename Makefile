PYTHON ?= python
CONFIG ?= configs/tasks/hover.toml

.PHONY: dev build clean test smoke benchmark rollout train eval compare

dev:
	$(PYTHON) -m pip install -e . --no-build-isolation

build:
	$(PYTHON) setup.py build_ext --inplace --force

clean:
	rm -rf build dist .pytest_cache src/flightrl/*.so src/flightrl/*.dylib src/flightrl/*.pyd
	find src -name "__pycache__" -type d -prune -exec rm -rf {} +

test:
	pytest

smoke:
	$(PYTHON) scripts/smoke_test.py --config $(CONFIG)

benchmark:
	$(PYTHON) scripts/benchmark_env.py --config $(CONFIG)

rollout:
	$(PYTHON) scripts/rollout_random.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

eval:
	$(PYTHON) scripts/eval.py --config $(CONFIG)

compare:
	$(PYTHON) scripts/compare_rewards.py
