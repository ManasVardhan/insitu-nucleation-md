# insitu-nucleation-md

Watching a liquid freeze: GPU-accelerated parallel molecular dynamics with in situ machine-learning
phase detection.

A Lennard-Jones liquid is quenched below its freezing point on GPUs (MPI + CUDA spatial decomposition).
A separate group of MPI analysis ranks labels every atom as liquid or crystal on the fly, using
Steinhardt bond-order features and a small neural network, and tracks the growing crystal nucleus
without writing the full trajectory to disk.

CSCI 596 (Scientific Computing and Visualization, USC) final project.

![A crystal growing from a seed in 1,048,576 supercooled LJ atoms](docs/media/nucleation_crystal.gif)

*1,048,576 atoms at T\* = 0.70: only the atoms that the in situ neural network labels crystalline are
drawn, colored by distance from the seed. Movies: [crystal](docs/media/nucleation_crystal.mp4),
[slab with both label sets](docs/media/nucleation_slab.mp4).*

- Results (performance, I/O, n_max(t), movie): [docs/results.md](docs/results.md)
- In situ pipeline and the in situ vs post hoc check: [docs/insitu.md](docs/insitu.md)
- Validation: [docs/validation.md](docs/validation.md), classifier: [docs/classifier.md](docs/classifier.md)
- Design: [docs/design.md](docs/design.md)
- Plan: [docs/plan.md](docs/plan.md)
