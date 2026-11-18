"""PE per atom during the constant-T hold of smoke-test logs.

Usage: python smoke_summary.py [--every 10000] runs/smoke/*/smoke.log
Crystallization at constant T releases latent heat, which shows up as a
drop in PE per atom. Prints PE at regular hold times relative to the
start of the hold (mean over the first 2000 hold steps after release).
"""
import argparse
import numpy as np

from energy_check import read_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--every", type=int, default=10000)
    a = ap.parse_args()
    for f in a.logs:
        p, d = read_log(f)
        step, T, pe = d[:, 0], d[:, 3], d[:, 4]
        s0 = p["smelt"] + p["sramp"]
        ref = (step > s0 + 1000) & (step <= s0 + 3000)
        pe0 = pe[ref].mean()
        hold = step > s0 + 1000
        out = []
        for s in range(s0 + a.every, int(step[-1]) + 1, a.every):
            k = np.argmin(np.abs(step - s))
            out.append(f"{(s - s0) // 1000}k:{pe[k] - pe0:+.4f}")
        print(f"{f}: Tq {p['tq']} <T> {T[hold].mean():.4f} PE0 {pe0:.4f} | dPE "
              + " ".join(out))


if __name__ == "__main__":
    main()
