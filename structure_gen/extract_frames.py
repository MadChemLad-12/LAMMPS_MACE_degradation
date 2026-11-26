#!/usr/bin/env python3
"""
extract_frames.py - pick frames from an MD trajectory and write them as plain .xyz files
(one file per frame) for CP2K single-point calculations.

Reads anything ASE can read: mace_md.py's traj.extxyz / events.extxyz / restart files,
extxyz from other codes, or a LAMMPS dump (--format lammps-dump-text --specorder Pt,C,O,H,F,S).
Atom order is never changed, so a CP2K result can be mapped back onto the source frame.

Selection modes (--mode)
  uniform  N evenly spaced frames (default, N from --n)
  stride   every frame after --start/--stop/--stride (no thinning beyond that)
  random   N random frames (--seed)
  fps      N frames by farthest-point sampling on a cheap Pt-environment descriptor
           (per-Pt coordination numbers to Pt, O, C, S; sorted) - picks structurally diverse
           frames, good for active-learning sets.
  events   frames around changes of the number of Pt atoms with <= --event-cn-max Pt
           neighbours within --pt-cut (default 0 -> "isolated Pt", i.e. detachment /
           dissolution-like events), +/- --window sampled frames; thinned to --n if more.

Filters applied before selection (frames failing them are skipped and counted)
  --min-dmin-ratio R   reject frames whose closest pair is < R x sum of covalent radii
                       (default 0.70: overlapping atoms give huge, useless DFT forces)
  --max-fmax F         reject frames whose stored MLIP max |F| exceeds F eV/A (needs forces
                       in the trajectory; ignored if absent)
  --where KEY:LO:HI    keep only frames whose extxyz info value KEY lies in [LO, HI]; repeat for
                       several keys. Needs info keys, e.g. from `analyze_run.py merge` (cn_ptpt,
                       cn_pto, dz_z, dS_min, ...). Example: frames near a barrier top:
                       --where cn_ptpt:2.0:3.5 --where cn_pto:0.5:2.5

Output (in --outdir)
  frame_00000_s<step>.xyz    plain xyz: "El x y z"; comment line holds step, time, source index,
                             cell vectors (CP2K ignores the comment; write the cell in your input)
  index.csv                  one row per frame: file, source index, md_step, time_ps, cell lengths
                             and angles, dmin_ratio, Pt_isolated, MLIP energy and max force (if stored)
  selected.extxyz            same frames as extxyz with cell + MLIP energy/forces, to merge with the
                             CP2K labels later (info: src_index, md_step)
  selected_all.xyz           all frames in one multi-frame xyz (only with --combined)

Examples
  python extract_frames.py run_300K/traj.extxyz --start 100 --mode uniform --n 50 --outdir sp_300K
  python extract_frames.py run/traj.extxyz --mode fps --n 100 --outdir sp_fps
  python extract_frames.py run/traj.extxyz --mode events --window 3 --n 60 --outdir sp_events
  python extract_frames.py dump.lammpstrj --format lammps-dump-text --specorder Pt,C,O,H,F,S \
        --mode uniform --n 40 --outdir sp_lmp
"""
import argparse
import csv
import os

import numpy as np
from ase.data import covalent_radii
from ase.io import iread, write
from ase.neighborlist import neighbor_list


def cell_vectors_str(at):
    return " ".join(f"{x:.6f}" for x in np.asarray(at.cell).reshape(-1))


def frame_metrics(at, need_env):
    """closest-approach ratio (to covalent radii sum) and, optionally, Pt environment."""
    sym = np.array(at.get_chemical_symbols())
    Z = at.get_atomic_numbers()
    rad = covalent_radii[Z]
    i, j, d = neighbor_list("ijd", at, 3.5)
    out = {"dmin_ratio": float(np.min(d / (rad[i] + rad[j]))) if len(d) else 9.9}
    isPt = sym == "Pt"
    out["env"], out["cn_pt"] = None, None
    if need_env and isPt.any():
        cols = []
        for els, cut in ((("Pt",), None), (("O",), 2.6), (("C",), 2.6), (("S",), 3.5)):
            m = np.isin(sym, els)
            c = 3.2 if cut is None else cut
            sel = isPt[i] & m[j] & (d < c)
            cols.append(np.bincount(i[sel], minlength=len(at))[isPt])
        env = np.column_stack(cols).astype(float)
        out["cn_pt"] = env[:, 0].astype(int)
        order = np.lexsort(env.T[::-1])     # sort Pt atoms by their environment -> permutation invariant
        out["env"] = env[order].ravel()
    return out


def energy_fmax(at):
    e = fm = None
    try:
        e = float(at.get_potential_energy())
    except Exception:
        pass
    try:
        fm = float(np.linalg.norm(at.get_forces(), axis=1).max())
    except Exception:
        pass
    return e, fm


def fps(X, n, first=0):
    X = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1.0)
    chosen = [first]
    dmin = np.linalg.norm(X - X[first], axis=1)
    while len(chosen) < min(n, len(X)):
        k = int(np.argmax(dmin))
        if dmin[k] <= 0:
            break
        chosen.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[k], axis=1))
    return sorted(chosen)


def write_plain_xyz(path, at, comment):
    with open(path, "w") as fh:
        fh.write(f"{len(at)}\n{comment}\n")
        for s, p in zip(at.get_chemical_symbols(), at.get_positions()):
            fh.write(f"{s:<2} {p[0]:16.8f} {p[1]:16.8f} {p[2]:16.8f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj")
    ap.add_argument("--outdir", default="extracted_frames")
    ap.add_argument("--mode", default="random", choices=["uniform", "stride", "random", "fps", "events"])
    ap.add_argument("--n", type=int, default=50, help="number of frames (uniform/random/fps; cap for events)")
    ap.add_argument("--start", type=int, default=0, help="first source frame (e.g. skip equilibration)")
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--format", default="extxyz", help="ASE format, e.g. extxyz, lammps-dump-text")
    ap.add_argument("--specorder", default=None, help="LAMMPS dump: element per atom type, e.g. Pt,C,O,H,F,S")
    ap.add_argument("--min-dmin-ratio", type=float, default=0.70)
    ap.add_argument("--max-fmax", type=float, default=None)
    ap.add_argument("--where", action="append", default=[], metavar="KEY:LO:HI",
                    help="keep frames with info[KEY] in [LO,HI] (repeatable)")
    ap.add_argument("--pt-cut", type=float, default=3.2, help="Pt-Pt neighbour cutoff for events/fps (A)")
    ap.add_argument("--event-cn-max", type=int, default=0, help="Pt with <= this many Pt neighbours counts as detached")
    ap.add_argument("--window", type=int, default=2, help="events: sampled frames kept before/after each event")
    ap.add_argument("--wrap", action="store_true", help="wrap atoms into the cell (default: keep as stored)")
    ap.add_argument("--combined", action="store_true", help="also write one multi-frame xyz")
    ap.add_argument("--seed", type=int, default=420)
    args = ap.parse_args()

    kw = {}
    if args.format:
        kw["format"] = args.format
    if args.specorder:
        kw["specorder"] = args.specorder.split(",")
    sl = f"{args.start}:{args.stop if args.stop is not None else ''}:{args.stride}"
    need_env = args.mode in ("fps", "events")

    # ---- pass 1: metrics only (no frames kept) ----
    recs, skipped = [], dict(overlap=0, fmax=0, natoms=0, where=0)
    n_seen, natoms0 = 0, None
    for k, at in enumerate(iread(args.traj, index=sl, **kw)):
        n_seen += 1
        src = args.start + k * args.stride
        if natoms0 is None:
            natoms0 = len(at)
        if len(at) != natoms0:
            skipped["natoms"] += 1
            continue
        ok = True
        for w in args.where:
            key, lo, hi = w.split(":")
            v = at.info.get(key)
            if v is None or not (float(lo) <= float(v) <= float(hi)):
                ok = False
                break
        if not ok:
            skipped["where"] += 1
            continue
        m = frame_metrics(at, need_env)
        e, fm = energy_fmax(at)
        if m["dmin_ratio"] < args.min_dmin_ratio:
            skipped["overlap"] += 1
            continue
        if args.max_fmax is not None and fm is not None and fm > args.max_fmax:
            skipped["fmax"] += 1
            continue
        m.update(src=src, k=k, step=at.info.get("md_step"), time=at.info.get("time_ps"), e=e, fmax=fm)
        if m["cn_pt"] is not None:
            m["n_iso"] = int((m["cn_pt"] <= args.event_cn_max).sum())
        recs.append(m)
    if not recs:
        raise SystemExit(f"no usable frames (read {n_seen}, skipped {skipped})")
    print(f"read {n_seen} frames, {len(recs)} pass the filters; skipped: {skipped}")

    # ---- selection ----
    N = len(recs)
    if args.mode == "stride":
        sel = list(range(N))
    elif args.mode == "uniform":
        sel = sorted(set(np.linspace(0, N - 1, min(args.n, N)).round().astype(int).tolist()))
    elif args.mode == "random":
        sel = sorted(np.random.default_rng(args.seed).choice(N, size=min(args.n, N), replace=False).tolist())
    elif args.mode == "fps":
        if recs[0]["env"] is None:
            raise SystemExit("fps needs Pt atoms in the system")
        lens = {len(r["env"]) for r in recs}
        if len(lens) != 1:
            raise SystemExit("Pt count changes between frames; fps descriptor undefined")
        sel = fps(np.array([r["env"] for r in recs]), args.n)
    else:  # events
        if recs[0]["cn_pt"] is None:
            raise SystemExit("events needs Pt atoms in the system")
        nis = np.array([r["n_iso"] for r in recs])
        ev = np.where(np.diff(nis) != 0)[0] + 1
        print(f"detached-Pt count changes at {len(ev)} sampled frames: {[recs[e]['src'] for e in ev[:20]]}"
              + (" ..." if len(ev) > 20 else ""))
        keep = set()
        for e in ev:
            keep.update(range(max(0, e - args.window), min(N, e + args.window + 1)))
        sel = sorted(keep)
        if not sel:
            raise SystemExit("no events found (try --event-cn-max 1, or a larger --pt-cut)")
        if len(sel) > args.n:
            sel = [sel[i] for i in np.linspace(0, len(sel) - 1, args.n).round().astype(int)]
    wanted = {recs[i]["src"]: pos for pos, i in enumerate(sel)}
    print(f"selected {len(sel)} frames (mode {args.mode})")

    # ---- pass 2: write ----
    os.makedirs(args.outdir, exist_ok=True)
    by_src = {recs[i]["src"]: recs[i] for i in sel}
    rows, ext_frames, comb = [], [], []
    for k, at in enumerate(iread(args.traj, index=sl, **kw)):
        src = args.start + k * args.stride
        if src not in wanted:
            continue
        m = by_src[src]
        if args.wrap:
            at.wrap()
        pos = wanted[src]
        step = m["step"] if m["step"] is not None else src
        name = f"frame_{pos:05d}_s{step}.xyz"
        a_, b_, c_, al, be, ga = at.cell.cellpar()
        t = f" time_ps={m['time']:.4f}" if m["time"] is not None else ""
        comment = f'step={step}{t} src_index={src} Lattice="{cell_vectors_str(at)}"'
        write_plain_xyz(os.path.join(args.outdir, name), at, comment)
        rows.append([name, src, step, "" if m["time"] is None else f"{m['time']:.4f}",
                     f"{a_:.4f}", f"{b_:.4f}", f"{c_:.4f}", f"{al:.2f}", f"{be:.2f}", f"{ga:.2f}",
                     f"{m['dmin_ratio']:.3f}", m.get("n_iso", ""),
                     "" if m["e"] is None else f"{m['e']:.6f}", "" if m["fmax"] is None else f"{m['fmax']:.4f}"])
        cp = at.copy()
        cp.info["src_index"], cp.info["md_step"] = int(src), int(step)
        ext_frames.append(cp)
        comb.append(at.copy())
    rows.sort(key=lambda r: r[0])
    with open(os.path.join(args.outdir, "index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "src_index", "md_step", "time_ps", "a", "b", "c", "alpha", "beta", "gamma",
                    "dmin_ratio", "n_detached_Pt", "E_mlip_eV", "Fmax_mlip_eV_A"])
        w.writerows(rows)
    write(os.path.join(args.outdir, "selected.extxyz"), ext_frames, format="extxyz")
    if args.combined:
        for i, at in enumerate(comb):
            write(os.path.join(args.outdir, "selected_all.xyz"), at, format="xyz", append=i > 0)
    print(f"wrote {len(rows)} xyz files + index.csv + selected.extxyz to {args.outdir}/")
    cells = {tuple(r[4:10]) for r in rows}
    if len(cells) > 1:
        print(f"note: cell varies between frames ({len(cells)} distinct) - check index.csv before copying one cell into every CP2K input")


if __name__ == "__main__":
    main()
