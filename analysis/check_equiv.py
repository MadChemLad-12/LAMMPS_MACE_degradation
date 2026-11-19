#!/usr/bin/env python3
"""
check_equiv.py - does the LAMMPS ML-IAP model give the same energies and forces as the ASE MACE
model you validated?  Run this BEFORE any production MD.

  python check_equiv.py --data data.lammps --ase-model MACE.model --lmp-model model-mliap_lammps.pt \
      --lmp "mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half" --device cuda

For the structure in --data (plus --n-rattled randomly displaced copies) it computes E and F
with ASE (MACECalculator) and with LAMMPS (`run 0`), using the same boundary as production
(p p f), and prints, per configuration:
  dE, dE/atom, max and RMS force difference, RMS difference per element.
Interpretation (rules of thumb, not standards):
  * float64 model in LAMMPS: force differences ~1e-6 eV/A or below.
  * float32 model: ~1e-4 - 1e-3 eV/A is normal. Much larger = a real mismatch.
  * dE can show a CONSTANT offset between ASE and LAMMPS (E0 handling) - harmless for MD, but
    find out why; if dE VARIES between the configurations, that is a real discrepancy.
  * If it fails: wrong head (multihead models: --head), wrong species order, model converted
    from a different checkpoint, or a different dtype.
--selftest swaps both sides for a Lennard-Jones potential to test this script itself.
"""
import argparse
import os
import re
import shlex
import subprocess
import tempfile

import numpy as np
from ase.calculators.lj import LennardJones
from ase.data import atomic_numbers
from ase.io import read, write


def lammps_ef(atoms, workdir, tag, lmp, pair_lines, specorder, env):
    workdir = os.path.abspath(workdir)
    data = os.path.join(workdir, f"{tag}.data")
    write(data, atoms, format="lammps-data", atom_style="atomic", specorder=specorder, masses=True)
    inp = os.path.join(workdir, f"{tag}.in")
    open(inp, "w").write(f"""units metal
boundary p p f
atom_style atomic
atom_modify map yes
newton on
read_data {data}
{pair_lines}
neighbor 1.0 bin
thermo_style custom step pe
dump d all custom 1 {tag}.dump id type x y z fx fy fz
dump_modify d sort id format float %.12g
run 0
print "FINAL_PE $(pe:%.12f)"
""")
    r = subprocess.run(shlex.split(lmp) + ["-in", inp, "-log", "none"], cwd=workdir, env=env,
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"FINAL_PE\s+(\S+)", out)
    if r.returncode != 0 or not m:
        raise SystemExit(f"LAMMPS failed for {tag} (exit {r.returncode}); command: {lmp} -in {inp}\nLast output:\n" + out[-2500:])
    lines = open(os.path.join(workdir, f"{tag}.dump")).read().split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    arr = np.array([[float(x) for x in l.split()] for l in lines[k + 1:] if l.strip()])
    return float(m.group(1)), arr[:, 2:5], arr[:, 5:8]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="LAMMPS data file (or an equilibrated snapshot)")
    ap.add_argument("--ase-model", help="MACE .model file for ASE")
    ap.add_argument("--lmp-model", help="converted model for LAMMPS (model-mliap_lammps.pt)")
    ap.add_argument("--lmp", default="lmp", help="LAMMPS launch command incl. Kokkos flags")
    ap.add_argument("--specorder", default="Pt,C,O,H,F,S")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", help="dtype of the ASE calculator")
    ap.add_argument("--head", default=None, help="head name for multihead models")
    ap.add_argument("--n-rattled", type=int, default=2)
    ap.add_argument("--rattle", type=float, default=0.05, help="Gaussian displacement sigma (A)")
    ap.add_argument("--ftol", type=float, default=1e-3, help="pass if max |dF| below this (eV/A)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--keep", default=None, help="directory to keep the LAMMPS files in")
    args = ap.parse_args()

    spec = args.specorder.split(",")
    atoms0 = read(args.data, format="lammps-data", atom_style="atomic",
                  Z_of_type={i + 1: atomic_numbers[e] for i, e in enumerate(spec)})
    atoms0.set_pbc((True, True, False))
    if args.selftest:
        calc = LennardJones(epsilon=0.01, sigma=2.2, rc=6.0, smooth=False)
        pair = "pair_style lj/cut 6.0\npair_coeff * * 0.01 2.2\npair_modify shift yes"
    else:
        if not (args.ase_model and args.lmp_model):
            raise SystemExit("give --ase-model and --lmp-model (or --selftest)")
        from mace.calculators import MACECalculator
        kw = dict(model_paths=args.ase_model, device=args.device, default_dtype=args.dtype)
        if args.head:
            kw["head"] = args.head
        calc = MACECalculator(**kw)
        pair = f"pair_style mliap unified {os.path.abspath(args.lmp_model)} 0\npair_coeff * * {' '.join(spec)}"

    cfgs = [("original", atoms0.copy())]
    for k in range(args.n_rattled):
        a = atoms0.copy()
        a.rattle(args.rattle, seed=100 + k)
        cfgs.append((f"rattled{k}", a))

    # ABSOLUTE path: LAMMPS runs with cwd=wd, so a relative --keep would make every path below point
    # to wd/wd/... ("Cannot open input script equiv_files/original.in")
    wd = os.path.abspath(args.keep or tempfile.mkdtemp(prefix="equiv_"))
    os.makedirs(wd, exist_ok=True)
    env = dict(os.environ)
    sym = np.array(atoms0.get_chemical_symbols())
    L = np.diag(atoms0.cell)[:2]
    dEs, worst = [], 0.0
    print(f"{'config':<10}{'dE (eV)':>14}{'dE/atom (meV)':>15}{'max|dF|':>12}{'rms dF':>12}{'rel max':>10}   rms dF by element")
    for tag, a in cfgs:
        a.calc = calc
        e_ase, f_ase = a.get_potential_energy(), a.get_forces()
        e_l, pos_l, f_l = lammps_ef(a, wd, tag, args.lmp, pair, spec, env)
        dpos = a.positions[:, :2] - pos_l[:, :2]
        dpos -= np.round(dpos / L) * L       # LAMMPS wraps x,y into the box
        if np.abs(dpos).max() > 1e-4 or np.abs(a.positions[:, 2] - pos_l[:, 2]).max() > 1e-4:
            raise SystemExit(f"{tag}: atom order/positions differ between ASE and LAMMPS (max {np.abs(dpos).max():.2e} A)")
        df = f_l - f_ase
        mx, rms = np.abs(df).max(), np.sqrt((df ** 2).mean())
        rel = mx / max(np.abs(f_ase).max(), 1e-12)
        per = " ".join(f"{e}:{np.sqrt((df[sym == e] ** 2).mean()):.1e}" for e in spec if (sym == e).any())
        dEs.append(e_l - e_ase)
        worst = max(worst, mx)
        print(f"{tag:<10}{e_l - e_ase:>14.6f}{1000 * (e_l - e_ase) / len(a):>15.4f}{mx:>12.2e}{rms:>12.2e}{rel:>10.1e}   {per}")
    spread = max(dEs) - min(dEs)
    print(f"\ndE spread across configurations: {spread:.2e} eV "
          f"({'constant offset only' if spread < 1e-4 else 'dE VARIES - real discrepancy, investigate'})")
    ok = worst < args.ftol or (args.selftest and worst / max(np.abs(f_ase).max(), 1e-12) < 1e-8)
    print(f"max |dF| over all configurations: {worst:.2e} eV/A -> {'PASS' if ok else 'FAIL'} (tolerance {args.ftol:g})")
    if not args.keep:
        print(f"LAMMPS files in {wd}")


if __name__ == "__main__":
    main()