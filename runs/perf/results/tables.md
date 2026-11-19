rows: 170 experiments: ['drift', 'insitu', 'insitu_sep', 'mixed', 'mixed_melt', 'scaling', 'sortfix', 'speedup']

### speedup

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
| cpu1 | 32000 | 56.401 | 0.195 | 2 | 55.611 |
| cpu1 | 108000 | 171.978 | 0.055 | 2 | 169.618 |
| cpu1 | 256000 | 413.708 | 0.026 | 2 | 408.026 |
| cpu1 | 500000 | 769.636 | 0.096 | 2 | 755.936 |
| cpu1 | 864000 | 1355.559 | 0.411 | 2 | 1327.085 |
| cpu10 | 32000 | 8.704 | 0.026 | 3 | 8.094 |
| cpu10 | 108000 | 27.478 | 0.081 | 3 | 26.316 |
| cpu10 | 256000 | 59.027 | 0.370 | 3 | 56.506 |
| cpu10 | 500000 | 112.738 | 0.032 | 3 | 108.510 |
| cpu10 | 864000 | 201.981 | 0.402 | 3 | 191.474 |
| cpu20 | 32000 | 4.587 | 0.028 | 3 | 4.111 |
| cpu20 | 108000 | 14.667 | 0.042 | 3 | 13.627 |
| cpu20 | 256000 | 30.583 | 0.036 | 3 | 29.052 |
| cpu20 | 500000 | 56.131 | 0.056 | 3 | 53.881 |
| cpu20 | 864000 | 99.367 | 0.492 | 3 | 95.528 |
| gpu1 | 32000 | 1.736 | 0.006 | 3 | 0.897 |
| gpu1 | 108000 | 4.770 | 0.008 | 3 | 2.385 |
| gpu1 | 256000 | 10.992 | 0.028 | 3 | 5.426 |
| gpu1 | 500000 | 23.924 | 0.080 | 3 | 10.151 |
| gpu1 | 864000 | 46.650 | 0.062 | 3 | 18.460 |
| gpu2 | 32000 | 1.326 | 0.002 | 3 | 0.728 |
| gpu2 | 108000 | 3.011 | 0.003 | 3 | 1.464 |
| gpu2 | 256000 | 6.023 | 0.008 | 3 | 2.790 |
| gpu2 | 500000 | 11.554 | 0.027 | 3 | 5.411 |
| gpu2 | 864000 | 21.695 | 0.062 | 3 | 9.242 |

### strong scaling

| run | P | step ms | std | speedup | efficiency |
|---|---|---|---|---|---|
| CPU path, N = 256,000 | 1 | 415.29 | 0.15 | 1.00 | 1.00 |
| CPU path, N = 256,000 | 2 | 225.14 | 0.17 | 1.84 | 0.92 |
| CPU path, N = 256,000 | 4 | 126.50 | 0.26 | 3.28 | 0.82 |
| CPU path, N = 256,000 | 8 | 75.19 | 0.02 | 5.52 | 0.69 |
| CPU path, N = 256,000 | 16 | 38.05 | 0.01 | 10.91 | 0.68 |
| CPU path, N = 256,000 | 20 | 30.38 | 0.19 | 13.67 | 0.68 |
| GPU path, N = 256,000 | 1 | 10.87 | 0.04 | 1.00 | 1.00 |
| GPU path, N = 256,000 | 2 | 6.55 | 0.06 | 1.66 | 0.83 |
| GPU path, N = 1,048,576 | 1 | 58.52 | 0.06 | 1.00 | 1.00 |
| GPU path, N = 1,048,576 | 2 | 33.78 | 0.04 | 1.73 | 0.87 |
| GPU path, N = 1,048,576 | 4 | 25.97 | 0.15 | 2.25 | 0.56 |

### weak scaling

| run | P | N | step ms | std | efficiency |
|---|---|---|---|---|---|
| CPU path, 32,000 atoms/core | 1 | 32,000 | 56.50 | 0.04 | 1.00 |
| CPU path, 32,000 atoms/core | 2 | 64,000 | 59.47 | 0.03 | 0.95 |
| CPU path, 32,000 atoms/core | 4 | 128,000 | 64.81 | 0.14 | 0.87 |
| CPU path, 32,000 atoms/core | 8 | 256,000 | 75.32 | 0.02 | 0.75 |
| CPU path, 32,000 atoms/core | 16 | 512,000 | 75.73 | 0.04 | 0.75 |
| CPU path, 32,000 atoms/core | 20 | 640,000 | 75.86 | 0.04 | 0.74 |
| GPU path, 262,144 atoms/GPU | 1 | 262,144 | 11.24 | 0.11 | 1.00 |
| GPU path, 262,144 atoms/GPU | 2 | 524,288 | 14.60 | 0.03 | 0.77 |

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

### analysis time vs K x step

| K | P_ana | 2nd node | K x step (s) | t_analysis mean (s) | t_analysis max (s) | stall total (s) |
|---|---|---|---|---|---|---|
| 50.0 | 4 | False | 1.36 | 0.83 | 1.10 | 0.000 |
| 100.0 | 1 | False | 3.13 | 3.08 | 3.15 | 13.641 |
| 100.0 | 2 | False | 2.65 | 1.56 | 1.61 | 0.000 |
| 100.0 | 4 | False | 2.67 | 0.83 | 0.85 | 0.000 |
| 100.0 | 4 | True | 2.58 | 1.00 | 1.03 | 0.000 |
| 200.0 | 4 | False | 5.29 | 0.83 | 0.85 | 0.000 |
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
