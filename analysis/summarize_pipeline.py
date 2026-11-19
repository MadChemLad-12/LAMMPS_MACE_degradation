#!/usr/bin/env python3
"""
summarize_pipeline.py RUNS_DIR - collect one row per system from a run_pipeline.sh output directory
(equivalence result, speed, temperature, structure flags, PLUMED pilot) and print/write summary.tsv.
"""
import csv
import os
import re
import sys

import numpy as np


def last_perf(log):
    if not os.path.exists(log):
        return None
    m = re.findall(r"Performance:\s+([\d.]+) ns/day,\s+[\d.]+ hours/ns,\s+([\d.]+) timesteps/s", open(log, errors="replace").read())
    return (float(m[-1][0]), float(m[-1][1])) if m else None


def thermo_T(log):
    if not os.path.exists(log):
        return None
    rows, hdr = [], None
    for line in open(log, errors="replace"):
        p = line.split()
        if len(p) >= 3 and p[0] == "Step" and "Temp" in p:
            hdr = p.index("Temp")
        elif hdr is not None and len(p) > hdr:
            try:
                rows.append(float(p[hdr]))
            except ValueError:
                pass
    return float(np.mean(rows[-max(5, len(rows) // 4):])) if rows else None


def main():
    root = sys.argv[1]
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "status.txt")):
            continue
        r = dict(system=d)
        r.update(dict(kv.split("=", 1) for kv in open(os.path.join(p, "status.txt")).read().split() if "=" in kv))
        eq = os.path.join(p, "equiv.log")
        if os.path.exists(eq):
            t = open(eq).read()
            m = re.search(r"max \|dF\| over all configurations: (\S+) eV/A -> (PASS|FAIL)", t)
            r["equiv"] = f"{m.group(2)} ({m.group(1)})" if m else "no result"
        perf = last_perf(os.path.join(p, "md.log"))
        if perf:
            r["steps_per_s"], r["ns_per_day"] = f"{perf[1]:.2f}", f"{perf[0]:.2f}"
        T = thermo_T(os.path.join(p, "md.log"))
        r["T_end_K"] = f"{T:.1f}" if T is not None else ""
        csvp = os.path.join(p, "md_traj_metrics.csv")
        if os.path.exists(csvp):
            rows = list(csv.DictReader(open(csvp)))
            if rows:
                g = lambda k: np.array([float(x[k]) for x in rows])
                r["Pt_detached_max"] = int(g("pt_detached").max())
                r["Nafion_broken_max"] = int(g("nafion_broken").max())
                r["min_S_Pt_A"] = f"{g('pt_s_min').min():.2f}"
                r["Pt_CN_end"] = f"{g('pt_cn_mean')[-1]:.2f}"
        out.append(r)
    if not out:
        sys.exit("no system directories with status.txt found")
    dry = os.path.exists(os.path.join(root, "DRY_RUN"))
    if dry:
        print("*** DRY RUN (dummy potential): speed, broken Nafion bonds, detached Pt and S-Pt distances below reflect the "
              "dummy potential, NOT the model. Only the 'md', 'pilot' and file columns tell you the plumbing works. ***\n")
    cols = ["system", "atoms", "build", "equiv", "md", "steps_per_s", "ns_per_day", "T_end_K", "Pt_detached_max",
            "Nafion_broken_max", "min_S_Pt_A", "Pt_CN_end", "pilot", "note"]
    with open(os.path.join(root, "summary.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in out:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    w = [max(len(c), *(len(str(r.get(c, ""))) for r in out)) for c in cols]
    print("  ".join(c.ljust(x) for c, x in zip(cols, w)))
    for r in out:
        print("  ".join(str(r.get(c, "")).ljust(x) for c, x in zip(cols, w)))
    bad = [r["system"] for r in out if r.get("md") not in ("ok",) or "FAIL" in r.get("equiv", "") or r.get("pilot") == "FAILED"]
    print(f"\n{len(out)} systems; problems: {bad if bad else 'none'}")
    print("Flags to read: Nafion_broken_max > 0 = unphysical bond breaking; Pt_detached_max > 0 = a Pt left the particle "
          "during plain MD (should not happen at 353 K without bias); steps_per_s is the number that sets the production budget.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()