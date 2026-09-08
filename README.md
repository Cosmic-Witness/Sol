# Sol
To produce pixel-precise instance segmentation masks for solar filaments observed in H-alpha images from the GONG (Global Oscillations Network Group) network.

Start with [the handoff](docs/HANDOFF.md) for measured experiment history and
[the architecture review](docs/architecture-review.md) for corrections and the
revised direction. [exp_032](experiments/exp_032_chunked/README.md) implements
checkpointed mask-loss chunks to reduce the full-resolution training workspace,
with differential tests and isolated memory probes. It has no new PQ result yet.
