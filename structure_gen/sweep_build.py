#!/usr/bin/env python3
"""
sweep_build.py - build many Pt/graphene/Nafion/water systems with build_interface.py WITHOUT
editing it. Put this file next to build_interface.py (structure_gen/).

  python sweep_build.py --out structures_sweep --sizes 0.8,1.0 --coverages 0.0,0.375,0.75 \
        --sites mixed --seeds 7,8 --heads graphene,pt --set n_nafion=2 --set nafion_xyz=/path/Simple_nafion.xyz

One output folder per condition:  <out>/cov0.375_mixed_s7_graphene/<shape>_<NNN>pt/data.lammps
(coverage 0.0 builds one bare system per head mode; site/seed are ignored there).
--set KEY=VALUE overrides any Config field (values are parsed as Python literals, else strings).
At the end <out>/manifest.txt lists the data.lammps files built BY THIS RUN (use --manifest-all for every
structure already under --out), one absolute path per line.
"""
import argparse
import ast
import itertools
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_interface as bi  # noqa: E402


def lit(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="structures_sweep")
    ap.add_argument("--sizes", default=None, help="comma list of volume-equivalent diameters (nm); default = Config.np_sizes_nm")
    ap.add_argument("--coverages", default="0.0")
    ap.add_argument("--sites", default="mixed")
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--heads", default=None, help="Nafion head modes, e.g. graphene,pt,free (default = Config.head_mode)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-packmol", action="store_true", help="reuse existing liquid.xyz (Config.run_packmol=False)")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--manifest-all", action="store_true",
                    help="list every data.lammps under --out, not only the ones built now")
    a = ap.parse_args()
    t0 = time.time() - 2.0

    names = {f.name for f in fields(bi.Config)}
    print(f"Parameters are: {names}")
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in names:
            sys.exit(f"--set {k}: not a field of Config")
        over[k] = lit(v)
    base = replace(bi.Config(), run_packmol=not a.no_packmol, view=a.view, **over)
    if a.sizes:
        base = replace(base, np_sizes_nm=tuple(float(x) for x in a.sizes.split(",")))
    covs = [float(x) for x in a.coverages.split(",")]
    sites = a.sites.split(",")
    seeds = [int(x) for x in a.seeds.split(",")]
    heads = a.heads.split(",") if a.heads else [base.head_mode]
    if any(c > 0 for c in covs):
        missing = [f for f in ("coverage", "ads_site", "ads_seed") if f not in names]
        if missing:
            sys.exit(f"Config has no field(s) {missing}: add the adsorbate fields to Config first")
    out = Path(a.out).resolve()
    done = set()
    print(f"output: {out}\n")
    for cov, site, seed, head in itertools.product(covs, sites, seeds, heads):
        if cov == 0.0:
            site, seed = "none", seeds[0]
        cond = f"cov{cov:.3f}_{site}_s{seed}_{head}"
        if cond in done:
            continue
        done.add(cond)
        kw = dict(head_mode=head, outdir=str(out / cond), seed=base.seed + seed)
        if "coverage" in names:
            kw["coverage"] = cov
        if cov > 0:
            kw.update(ads_site=site, ads_seed=seed)
        cfg = replace(base, **kw)
        print(f"=== {cond}")
        for d in cfg.np_sizes_nm:
            bi.build_one(cfg, d)
        print()
    found = sorted(p for p in out.glob("*/*/data.lammps") if a.manifest_all or p.stat().st_mtime >= t0)
    (out / "manifest.txt").write_text("".join(f"{p}\n" for p in found))
    print(f"{len(found)} structures built now; manifest: {out / 'manifest.txt'}")


if __name__ == "__main__":
    main()