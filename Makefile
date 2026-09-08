# 4090 data plane entry points. All targets use the project conda env.
PY := /data5/ljt_data/conda_envs/embodiedflow/bin/python
export PYTHONPATH := src

# real-data entry points (used once 5090 Isaac episodes arrive)
EPISODE ?= samples/episode_v0
DATA ?= data/real_episodes

.PHONY: test dataset-smoke overfit ddp-smoke benchmark quality-report manifest real-overfit

test:
	$(PY) -m pytest -q

dataset-smoke:
	$(PY) -m embodiedflow.dataplane.smoke --chunk-len 20 --batch-size 4

overfit:
	$(PY) -m embodiedflow.dataplane.train --gpu auto --run-id overfit-fixture-v0 --steps 400

ddp-smoke:
	$(PY) tools/launch_ddp.py 2 --warmup-steps 5 --measure-steps 15 --label ddp-smoke

benchmark:
	$(PY) tools/run_benchmark.py --warmup-steps 20 --measure-steps 100

quality-report:
	$(PY) tools/episode_quality_report.py $(EPISODE)

manifest:
	$(PY) tools/episode_manifest.py $(DATA) --out artifacts/manifest/$(notdir $(DATA))

real-overfit:
	$(PY) -m embodiedflow.dataplane.train --fixture $(DATA) --gpu auto \
		--run-id real-overfit-v0 --steps 1000 --pool-size 8
