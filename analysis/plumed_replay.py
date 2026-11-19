#!/usr/bin/env python3
"""
plumed_replay.py - run a PLUMED input over frames you ALREADY have, with no MD and no MACE.

Use it to check a plumed.dat (atom groups, units, CV values, sign of the height, wall position)
on the real equilibrated structure/trajectory before spending GPU time:

  python plumed_replay.py pipeline_runs/<run>/<system>/md.extxyz pilot/plumed.dat --stride 5
  python plumed_replay.py ... --plumed-bin /path/to/plumed

It writes frames.xyz (plain xyz) into --workdir, runs `plumed driver` (A and eV conventions as in the
PLUMED files from prep_plumed.py) and prints, for every CV in COLVAR: first, mean, min, max.
It then flags the usual mistakes. A METAD in the file just deposits hills along the replayed frames
(the frames are not biased), which is fine for a syntax/units check.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
from ase.io import iread


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", help="extxyz / lammpstrj / anything ASE reads")
    ap.add_argument("plumed")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--dt-ps", type=float, default=0.0005, help="MD timestep of the original run (ps)")
    ap.add_argument("--dump-interval", type=int, default=200, help="MD steps between stored frames")
    ap.add_argument("--temp", type=float, default=353.0)
    ap.add_argument("--plumed-bin", default="plumed")
    ap.add_argument("--workdir", default="plumed_replay")
    ap.add_argument("--format", default=None)
    a = ap.parse_args()
    if shutil.which(a.plumed_bin) is None and not os.path.isfile(a.plumed_bin):
        sys.exit(f"cannot find '{a.plumed_bin}' (use --plumed-bin)")
    os.makedirs(a.workdir, exist_ok=True)
    kw = {"format": a.format} if a.format else {}
    n, cell = 0, None
    with open(os.path.join(a.workdir, "frames.xyz"), "w") as fh:
        for k, at in enumerate(iread(a.frames, index=f"::{a.stride}", **kw)):
            if n >= a.max_frames:
                break
            if cell is None:
                cell = np.diag(at.cell)
            fh.write(f"{len(at)}\nframe {k}\n")
            for s, p in zip(at.get_chemical_symbols(), at.get_positions()):
                fh.write(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            n += 1
    if not n:
        sys.exit("no frames read")
    shutil.copy(a.plumed, os.path.join(a.workdir, "plumed.dat"))
    os.makedirs(os.path.join(a.workdir, "walkers"), exist_ok=True)
    kT = 0.0083144626 * a.temp * 1.0     # kJ/mol, what plumed driver expects
    dt = a.dt_ps * a.dump_interval * a.stride
    cmd = [a.plumed_bin, "driver", "--plumed", "plumed.dat", "--ixyz", "frames.xyz",
           "--box", f"{cell[0]},{cell[1]},{cell[2]}", "--length-units", "0.1", "--timestep", f"{dt}", "--kt", f"{kT:.4f}"]
    print(f"{n} frames, box {cell.round(2)} A -> running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=a.workdir, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(a.workdir, "COLVAR")):
        print((r.stdout + r.stderr)[-2500:])
        sys.exit("plumed driver failed (see above): the PLUMED file or the atom numbering has a problem")
    names, rows = None, []
    for line in open(os.path.join(a.workdir, "COLVAR")):
        if line.startswith("#! FIELDS"):
            names = line.split()[2:]
        elif not line.startswith("#"):
            rows.append([float(x) for x in line.split()])
    d = dict(zip(names, np.array(rows).T))
    print(f"\n{'CV':<12}{'first':>10}{'mean':>10}{'min':>10}{'max':>10}")
    for k in names[1:]:
        v = d[k]
        print(f"{k:<12}{v[0]:>10.3f}{v.mean():>10.3f}{v.min():>10.3f}{v.max():>10.3f}")
    warn = []
    if any(not np.isfinite(d[k]).all() for k in names):
        warn.append("NaN/inf in COLVAR")
    if "dz.z" in d and d["dz.z"].mean() < 0:
        warn.append("dz.z is negative: the height is measured the wrong way round (use DISTANCE ATOMS=graphene,target)")
    if "cn_ptpt" in d and d["cn_ptpt"].min() < 0.5:
        warn.append("cn_ptpt < 0.5 in a replay of a plain MD trajectory: the target has already left the particle, or the Pt group is wrong")
    if "cn_pto" in d and d["cn_pto"].max() < 0.05:
        warn.append("cn_pto is ~0 everywhere: no O within 2.7 A of the target (fine for bare Pt in vacuum-like sites, suspicious if the target is solvated)")
    print("\nCHECKS: " + ("; ".join(warn) if warn else "no obvious problems") +
          "\nAlso compare dz.z / dC.min with the height and Pt-C distance printed by prep_plumed.py for the target atom.")


if __name__ == "__main__":
    main()