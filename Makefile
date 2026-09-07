# 4090 data plane entry points. All targets use the project conda env.
PY := /data5/ljt_data/conda_envs/embodiedflow/bin/python
export PYTHONPATH := src

.PHONY: test dataset-smoke overfit ddp-smoke benchmark

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
