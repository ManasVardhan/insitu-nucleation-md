rows: 297 experiments: ['drift', 'insitu', 'insitu_sep', 'mixed', 'mixed_melt', 'scaling', 'sortfix', 'speedup']

### speedup cell-order kernel

| N | CPU 1 core | CPU 10 cores | CPU 20 cores | 1 P100 | 2 P100 | 1 P100 vs 1 core | 1 P100 vs node | 2 P100 vs node |
|---|---|---|---|---|---|---|---|---|
| 32,000 | 56.32 | 8.73 | 4.61 | 1.56 | 1.11 | 36.0x | 2.9x | 4.2x |
| 108,000 | 171.98 | 27.49 | 14.65 | 4.33 | 2.76 | 39.7x | 3.4x | 5.3x |
| 256,000 | 413.85 | 58.70 | 30.43 | 9.67 | 5.54 | 42.8x | 3.1x | 5.5x |
| 500,000 | 772.32 | 112.59 | 56.09 | 21.68 | 10.39 | 35.6x | 2.6x | 5.4x |
| 864,000 | 1358.82 | 201.85 | 99.41 | 41.86 | 19.67 | 32.5x | 2.4x | 5.1x |

### speedup host-order kernel

| N | CPU 1 core | CPU 10 cores | CPU 20 cores | 1 P100 | 2 P100 | 1 P100 vs 1 core | 1 P100 vs node | 2 P100 vs node |
|---|---|---|---|---|---|---|---|---|
| 32,000 | 56.40 | 8.70 | 4.59 | 1.74 | 1.33 | 32.5x | 2.6x | 3.5x |
| 108,000 | 171.98 | 27.48 | 14.67 | 4.77 | 3.01 | 36.1x | 3.1x | 4.9x |
| 256,000 | 413.71 | 59.03 | 30.58 | 10.99 | 6.02 | 37.6x | 2.8x | 5.1x |
| 500,000 | 769.64 | 112.74 | 56.13 | 23.92 | 11.55 | 32.2x | 2.3x | 4.9x |
| 864,000 | 1355.56 | 201.98 | 99.37 | 46.65 | 21.69 | 29.1x | 2.1x | 4.6x |

(per-rank std)

| config | N | step ms | std | n | force ms |
|---|---|---|---|---|---|
| cpu1 | 32000 | 56.319 | 0.100 | 2 | 55.532 |
| cpu1 | 108000 | 171.983 | 0.001 | 2 | 169.621 |
| cpu1 | 256000 | 413.847 | 0.211 | 2 | 408.158 |
| cpu1 | 500000 | 772.320 | 0.168 | 2 | 758.632 |
| cpu1 | 864000 | 1358.823 | 1.505 | 2 | 1330.522 |
| cpu10 | 32000 | 8.732 | 0.070 | 3 | 8.120 |
| cpu10 | 108000 | 27.488 | 0.133 | 3 | 26.286 |
| cpu10 | 256000 | 58.701 | 0.039 | 3 | 56.465 |
| cpu10 | 500000 | 112.589 | 0.078 | 3 | 108.390 |
| cpu10 | 864000 | 201.845 | 0.244 | 3 | 191.369 |
| cpu20 | 32000 | 4.606 | 0.014 | 3 | 4.115 |
| cpu20 | 108000 | 14.648 | 0.003 | 3 | 13.623 |
| cpu20 | 256000 | 30.432 | 0.090 | 3 | 29.181 |
| cpu20 | 500000 | 56.086 | 0.184 | 3 | 53.838 |
| cpu20 | 864000 | 99.410 | 0.698 | 3 | 95.505 |
| gpu1 | 32000 | 1.563 | 0.003 | 3 | 0.732 |
| gpu1 | 108000 | 4.333 | 0.005 | 3 | 1.951 |
| gpu1 | 256000 | 9.672 | 0.030 | 3 | 4.158 |
| gpu1 | 500000 | 21.685 | 0.260 | 3 | 7.877 |
| gpu1 | 864000 | 41.862 | 0.153 | 3 | 13.659 |
| gpu2 | 32000 | 1.105 | 0.004 | 3 | 0.529 |
| gpu2 | 108000 | 2.759 | 0.005 | 3 | 1.211 |
| gpu2 | 256000 | 5.540 | 0.028 | 3 | 2.297 |
| gpu2 | 500000 | 10.391 | 0.080 | 3 | 4.240 |
| gpu2 | 864000 | 19.670 | 0.122 | 3 | 7.128 |

### strong scaling

| run | P | step ms | std | speedup | efficiency |
|---|---|---|---|---|---|
| CPU path, N = 256,000 | 1 | 416.86 | 0.29 | 1.00 | 1.00 |
| CPU path, N = 256,000 | 2 | 224.84 | 0.57 | 1.85 | 0.93 |
| CPU path, N = 256,000 | 4 | 126.34 | 0.16 | 3.30 | 0.82 |
| CPU path, N = 256,000 | 8 | 75.37 | 0.53 | 5.53 | 0.69 |
| CPU path, N = 256,000 | 16 | 38.01 | 0.00 | 10.97 | 0.69 |
| CPU path, N = 256,000 | 20 | 30.29 | 0.10 | 13.76 | 0.69 |
| GPU path, N = 256,000 | 1 | 9.69 | 0.09 | 1.00 | 1.00 |
| GPU path, N = 256,000 | 2 | 6.08 | 0.01 | 1.60 | 0.80 |
| GPU path, N = 1,048,576 | 1 | 51.73 | 0.21 | 1.00 | 1.00 |
| GPU path, N = 1,048,576 | 2 | 30.50 | 0.20 | 1.70 | 0.85 |
| GPU path, N = 1,048,576 | 4 | 23.58 | 0.11 | 2.19 | 0.55 |
| GPU path, N = 1,048,576, before the fix | 1 | 58.52 | 0.06 | 1.00 | 1.00 |
| GPU path, N = 1,048,576, before the fix | 2 | 33.78 | 0.04 | 1.73 | 0.87 |
| GPU path, N = 1,048,576, before the fix | 4 | 25.97 | 0.15 | 2.25 | 0.56 |

### weak scaling

| run | P | N | step ms | std | efficiency |
|---|---|---|---|---|---|
| CPU path, 32,000 atoms/core | 1 | 32,000 | 56.63 | 0.03 | 1.00 |
| CPU path, 32,000 atoms/core | 2 | 64,000 | 59.46 | 0.16 | 0.95 |
| CPU path, 32,000 atoms/core | 4 | 128,000 | 64.73 | 0.18 | 0.87 |
| CPU path, 32,000 atoms/core | 8 | 256,000 | 75.20 | 0.07 | 0.75 |
| CPU path, 32,000 atoms/core | 16 | 512,000 | 76.10 | 0.47 | 0.74 |
| CPU path, 32,000 atoms/core | 20 | 640,000 | 75.88 | 0.26 | 0.75 |
| GPU path, 262,144 atoms/GPU | 1 | 262,144 | 9.96 | 0.08 | 1.00 |
| GPU path, 262,144 atoms/GPU | 2 | 524,288 | 13.61 | 0.08 | 0.73 |
| GPU path, 262,144 atoms/GPU, before the fix | 1 | 262,144 | 11.24 | 0.11 | 1.00 |
| GPU path, 262,144 atoms/GPU, before the fix | 2 | 524,288 | 14.60 | 0.03 | 0.77 |

### in situ overhead

| job | run | n | SIM ms/step | std | vs off (%) | +- | t_analysis mean (s) | t_analysis max (s) | stall total (s) | stall max (s) | SIM halo+pack (ms/step) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| insitu | off | 6 | 25.08 | 0.05 | +0.0 | 0.3 | nan | nan | nan | nan | nan |
| insitu | K=50, P_ana=4 | 3 | 26.47 | 0.03 | +5.6 | 0.2 | 0.834 | 1.102 | 0.000 | 0.0000 | 0.143 |
| insitu | K=100, P_ana=4 | 3 | 26.07 | 0.10 | +4.0 | 0.4 | 0.829 | 0.855 | 0.000 | 0.0000 | 0.076 |
| insitu | K=200, P_ana=4 | 3 | 25.82 | 0.03 | +3.0 | 0.2 | 0.828 | 0.850 | 0.000 | 0.0000 | 0.040 |
| insitu | K=100, P_ana=2 | 3 | 26.10 | 0.03 | +4.1 | 0.2 | 1.561 | 1.611 | 0.000 | 0.0000 | 0.077 |
| insitu | K=100, P_ana=1 | 3 | 31.39 | 0.11 | +25.2 | 0.5 | 3.080 | 3.153 | 13.641 | 0.5559 | 0.080 |
| insitu_sep | off | 3 | 25.13 | 0.06 | +0.0 | 0.3 | nan | nan | nan | nan | nan |
| insitu_sep | K=100, P_ana=4 | 3 | 26.05 | 0.06 | +3.7 | 0.3 | 0.824 | 0.847 | 0.000 | 0.0000 | 0.075 |
| insitu_sep | K=100, P_ana=4, ANA on a 2nd node | 3 | 25.18 | 0.05 | +0.2 | 0.3 | 1.001 | 1.025 | 0.000 | 0.0000 | 0.072 |
| insitu_cell | off | 3 | 18.61 | 0.19 | +0.0 | 1.5 | nan | nan | nan | nan | nan |
| insitu_cell | K=100, P_ana=4 | 3 | 19.58 | 0.06 | +5.2 | 1.1 | 0.836 | 0.860 | 0.000 | 0.0000 | 0.075 |

### analysis time vs K x step

| K | P_ana | 2nd node | cell-order kernel | K x step (s) | t_analysis mean (s) | t_analysis max (s) | stall total (s) |
|---|---|---|---|---|---|---|---|
| 50.0 | 4 | False | False | 1.36 | 0.83 | 1.10 | 0.000 |
| 100.0 | 1 | False | False | 3.13 | 3.08 | 3.15 | 13.641 |
| 100.0 | 2 | False | False | 2.65 | 1.56 | 1.61 | 0.000 |
| 100.0 | 4 | False | False | 2.67 | 0.83 | 0.85 | 0.000 |
| 100.0 | 4 | False | True | 1.96 | 0.84 | 0.86 | 0.000 |
| 100.0 | 4 | True | False | 2.58 | 1.00 | 1.03 | 0.000 |
| 200.0 | 4 | False | False | 5.29 | 0.83 | 0.85 | 0.000 |
drift host-order threads off mean ms/step melt 27.54 ramp 31.35 hold 31.79 | force melt 14.41 hold 19.32
drift host-order threads on mean ms/step melt 28.60 ramp 32.50 hold 32.92 | force melt 16.53 hold 20.91
drift cell-order threads off mean ms/step melt 18.87 ramp 19.06 hold 18.72 | force melt 7.83 hold 7.92
drift cell-order threads on mean ms/step melt 19.97 ramp 20.36 hold 20.10 | force melt 7.78 hold 7.94

### mixed memory order

| state | layout | N | steps | first 500 (ms) | last 500 (ms) | force first | force last |
|---|---|---|---|---|---|---|---|
| solid (NVE from lattice) | cpu20 | 256000 | 3000 | 30.53 | 30.53 | 29.11 | 28.91 |
| solid (NVE from lattice) | gpu1_old | 256000 | 3000 | 10.98 | 11.01 | 5.44 | 5.46 |
| solid (NVE from lattice) | gpu1_new | 256000 | 3000 | 9.67 | 9.67 | 4.14 | 4.14 |
| solid (NVE from lattice) | gpu2_old | 256000 | 3000 | 6.59 | 6.58 | 2.85 | 2.84 |
| solid (NVE from lattice) | gpu2_new | 256000 | 3000 | 6.09 | 6.10 | 2.47 | 2.44 |
| solid (NVE from lattice) | cpu20 | 864000 | 2000 | 100.39 | 99.96 | 95.61 | 95.33 |
| solid (NVE from lattice) | gpu1_old | 864000 | 2000 | 46.82 | 46.80 | 18.47 | 18.56 |
| solid (NVE from lattice) | gpu1_new | 864000 | 2000 | 41.73 | 41.81 | 13.64 | 13.69 |
| solid (NVE from lattice) | gpu2_old | 864000 | 2000 | 27.11 | 27.21 | 9.84 | 9.96 |
| solid (NVE from lattice) | gpu2_new | 864000 | 2000 | 24.61 | 24.59 | 7.25 | 7.25 |
| liquid (T* = 2 thermostat) | cpu20 | 256000 | 6000 | 30.90 | 32.72 | 29.18 | 31.19 |
| liquid (T* = 2 thermostat) | gpu1_old | 256000 | 6000 | 11.98 | 16.07 | 5.82 | 9.52 |
| liquid (T* = 2 thermostat) | gpu1_new | 256000 | 6000 | 10.52 | 11.03 | 4.40 | 4.48 |
| liquid (T* = 2 thermostat) | gpu2_old | 256000 | 6000 | 7.05 | 9.43 | 3.04 | 5.17 |
| liquid (T* = 2 thermostat) | gpu2_new | 256000 | 6000 | 6.32 | 6.62 | 2.38 | 2.40 |
