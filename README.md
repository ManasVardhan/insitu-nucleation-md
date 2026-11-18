# insitu-nucleation-md

Watching a liquid freeze: GPU-accelerated parallel molecular dynamics with in situ machine-learning
phase detection.

A Lennard-Jones liquid is quenched below its freezing point on GPUs (MPI + CUDA spatial decomposition).
A separate group of MPI analysis ranks labels every atom as liquid or crystal on the fly, using
Steinhardt bond-order features and a small neural network, and tracks the growing crystal nucleus
without writing the full trajectory to disk.

CSCI 596 (Scientific Computing and Visualization, USC) final project. Work in progress.

- Design: [docs/design.md](docs/design.md)
- Plan: [docs/plan.md](docs/plan.md)
