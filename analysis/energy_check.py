"""NVE energy drift and thermostat settling from a pmd log.

Usage: python energy_check.py run.log
Reads the protocol line and the step table printed by pmd / pmd_cuda.
"""
import re
import sys
import numpy as np


def read_log(path):
    proto, rows = None, []
    for line in open(path):
        if line.startswith("Protocol:"):
            m = re.search(r"melt (\S+) x (\d+) \| ramp (\d+) \| hold (\S+) x (\d+) "
                          r"\| nve (\d+)", line)
            proto = dict(tmelt=float(m[1]), smelt=int(m[2]), sramp=int(m[3]),
                         tq=float(m[4]), shold=int(m[5]), snve=int(m[6]))
        f = line.split()
        if len(f) == 7 and f[0].isdigit():
            rows.append([float(x) for x in f])
    return proto, np.array(rows)


def main(path):
    p, d = read_log(path)
    step, T, pe, te = d[:, 0], d[:, 3], d[:, 4], d[:, 6]
    s0 = p["smelt"] + p["sramp"]
    s1 = s0 + p["shold"]

    # Thermostat: T during the NVT hold after the quench ramp
    hold = (step > s0) & (step <= s1)
    within = np.abs(T[hold] / p["tq"] - 1.0) < 0.01
    # first step after which the 100-step running mean stays within 1%
    w = max(1, int(100 / (step[1] - step[0])))
    rm = np.convolve(T[hold], np.ones(w) / w, mode="valid")
    ok = np.abs(rm / p["tq"] - 1.0) < 0.01
    bad = np.where(~ok)[0]
    settle = step[hold][bad[-1] + w] - s0 if len(bad) else 0
    late = hold & (step > s0 + p["shold"] / 2)
    print(f"thermostat target {p['tq']}: mean T (2nd half of hold) "
          f"{T[late].mean():.5f} (rel dev {T[late].mean() / p['tq'] - 1:+.2e}), "
          f"std {T[late].std():.5f}; instantaneous within 1%: "
          f"{within.mean() * 100:.1f}% of samples; running mean settled "
          f"within 1% after {settle:.0f} steps")

    # NVE: total energy per atom over the thermostat-free segment
    if p["snve"] > 0:
        nve = step >= s1
        e, s = te[nve], step[nve]
        slope = np.polyfit(s - s[0], e, 1)[0]
        print(f"NVE {p['snve']} steps: E0 {e[0]:.8f}, E_end - E0 {e[-1] - e[0]:+.3e}, "
              f"max |E - E0| {np.abs(e - e[0]).max():.3e}, fit drift over "
              f"{p['snve']} steps {slope * p['snve']:+.3e}, std {e.std():.3e} "
              f"(per atom); mean T {T[nve].mean():.4f}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f"== {f}")
        main(f)
