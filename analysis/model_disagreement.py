#!/usr/bin/env python3
"""
model_disagreement.py - cheap accuracy screen: how much do two differently trained MACE models disagree on the
frames of YOUR runs? No DFT needed.

  python model_disagreement.py --models mace_V9_active_learning_stagetwo.model mace_V11_active_learning.model \
        --frames pipeline_runs/<run>/<system>/md.extxyz --n 30 --out disagree_sysA

Disagreement is a WARNING, not an error bar: two models trained on overlapping data can agree and both be wrong.
Large disagreement, however, marks configurations that are certainly not well described. Per frame it reports the RMS
force difference overall, per element, and for the atoms that matter here (within --near A of any Pt, adsorbed O,
sulfonate S/O), plus the energy difference per atom after removing a constant offset between the two models.
Output: <out>_frames.csv, <out>_disagree.extxyz (frames with info key model_dF_rms, so that
`extract_frames.py --where model_dF_rms:0.08:99` picks the worst ones for DFT) and a printed summary.
--selftest replaces the two models by two Lennard-Jones parameter sets, to test the script itself.
"""
import argparse
import csv

import numpy as np
from ase.calculators.lj import LennardJones
from ase.io import iread, write
from ase.neighborlist import neighbor_list


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs=2)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--n", type=int, default=30, help="number of frames, evenly spaced")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--near", type=float, default=4.0, help="radius around Pt for the 'reaction zone' statistics (A)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--head", default=None)
    ap.add_argument("--out", default="disagree")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    frames = list(iread(a.frames, index=f"{a.start}:"))
    sel = np.unique(np.linspace(0, len(frames) - 1, min(a.n, len(frames))).round().astype(int))
    frames = [frames[i] for i in sel]
    if a.selftest:
        calcs = [LennardJones(epsilon=0.010, sigma=2.2, rc=6.0), LennardJones(epsilon=0.011, sigma=2.2, rc=6.0)]
    else:
        if not a.models:
            raise SystemExit("give --models A.model B.model (or --selftest)")
        from mace.calculators import MACECalculator
        kw = dict(device=a.device, default_dtype=a.dtype)
        if a.head:
            kw["head"] = a.head
        calcs = [MACECalculator(model_paths=m, **kw) for m in a.models]

    rows, dE = [], []
    elems = sorted(set(frames[0].get_chemical_symbols()))
    for k, at in enumerate(frames):
        at = at.copy()
        out = []
        for c in calcs:
            at.calc = c
            out.append((at.get_potential_energy(), at.get_forces().copy()))
        (e1, f1), (e2, f2) = out
        d = f1 - f2
        sym = np.array(at.get_chemical_symbols())
        n = np.linalg.norm(d, axis=1)
        isPt = sym == "Pt"
        near = np.zeros(len(at), bool)
        if isPt.any():
            i, j, dist = neighbor_list("ijd", at, a.near)
            near[i[isPt[j]]] = True
            near |= isPt
        r = dict(frame=int(sel[k]) + a.start, step=at.info.get("Timestep", ""), dE_per_atom_meV=1000 * (e1 - e2) / len(at),
                 dF_rms=float(np.sqrt((d ** 2).mean())), dF_max=float(n.max()),
                 dF_rms_near_Pt=float(np.sqrt((d[near] ** 2).mean())) if near.any() else np.nan)
        for el in elems:
            r[f"dF_rms_{el}"] = float(np.sqrt((d[sym == el] ** 2).mean()))
        rows.append(r)
        dE.append(r["dE_per_atom_meV"])
        at.info["model_dF_rms"] = r["dF_rms"]
        at.info["model_dF_rms_near_Pt"] = r["dF_rms_near_Pt"]
        frames[k] = at
    dE = np.array(dE)
    for r in rows:
        r["dE_minus_offset_meV"] = r["dE_per_atom_meV"] - dE.mean()
    with open(a.out + "_frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write(a.out + "_disagree.extxyz", frames, format="extxyz")
    g = lambda k: np.array([r[k] for r in rows], float)
    print(f"{len(rows)} frames | models: {a.models if not a.selftest else 'LJ self-test'}")
    print(f"force disagreement (eV/A): overall rms {np.sqrt((g('dF_rms') ** 2).mean()):.4f} | worst frame rms {g('dF_rms').max():.4f}"
          f" | worst single atom {g('dF_max').max():.3f}")
    print(f"near Pt (<{a.near} A): rms {np.sqrt(np.nanmean(g('dF_rms_near_Pt') ** 2)):.4f}")
    print("per element rms (eV/A): " + "  ".join(f"{el} {np.sqrt((g('dF_rms_' + el) ** 2).mean()):.4f}" for el in elems))
    print(f"energy: constant offset {dE.mean():.2f} meV/atom (ignored); frame-to-frame spread {np.std(dE):.3f} meV/atom")
    worst = np.argsort(g("dF_rms"))[::-1][:5]
    print("worst frames (frame index, step, rms dF): " + ", ".join(f"{rows[i]['frame']}/{rows[i]['step']}/{rows[i]['dF_rms']:.3f}" for i in worst))
    print(f"wrote {a.out}_frames.csv and {a.out}_disagree.extxyz")


if __name__ == "__main__":
    main()
