"""4090 data plane: dataset, training, checkpoint export.

Implements the frozen contract-v0.1 episode interface as the only data
format. See docs/4090/README.md for entry points.

Submodules are intentionally NOT imported here: the checkpoint load path
(5090 side) only needs ``model`` + ``config`` (torch + pyyaml), and an
eager import would transitively require dataset/stats (numpy, jsonschema).
Import submodules explicitly, e.g.
``from embodiedflow.dataplane.model import ChunkedActionTransformerMinimal``.
"""
