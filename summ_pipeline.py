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
    """mean temperature over the second half of the LAST thermo block only (stops at 'Loop time', so the timing and
    histogram tables LAMMPS prints afterwards are not read as data)"""
    if not os.path.exists(log):
        return None
    hdr, cur, last = None, None, None
    for line in open(log, errors="replace"):
        p = line.split()
        if hdr is None:
            if len(p) >= 3 and p[0] == "Step" and "Temp" in p:
                hdr, cur = p.index("Temp"), []
            continue
        if line.startswith("Loop time") or line.startswith("ERROR"):
            if cur:
                last = cur
            hdr, cur = None, None
            continue
        if len(p) > hdr:
            try:
                cur.append(float(p[hdr]))
            except ValueError:
                pass
    if cur:
        last = cur
    return float(np.mean(last[len(last) // 2:])) if last else None


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
                h = len(rows) // 2
                r["Pt_detached_max"] = int(g("pt_detached").max())
                if "nafion_broken_OH" in rows[0]:
                    r["SO3H_dissoc_max"] = int(g("nafion_broken_OH").max())
                    r["backbone_broken_max"] = int(g("nafion_broken_other").max())
                else:
                    r["backbone_broken_max"] = int(g("nafion_broken").max())   # old CSV: O-H not separated
                r["S_Pt_contact_frac"] = f"{(g('pt_s_min')[h:] < 3.6).mean():.2f}"
                r["Os_on_Pt_mean"] = f"{g('n_Os_on_Pt')[h:].mean():.1f}"
                r["Pt_CN_mean"] = f"{g('pt_cn_mean')[h:].mean():.2f}"
        out.append(r)
    if not out:
        sys.exit("no system directories with status.txt found")
    dry = os.path.exists(os.path.join(root, "DRY_RUN"))
    if dry:
        print("*** DRY RUN (dummy potential): speed, broken Nafion bonds, detached Pt and S-Pt distances below reflect the "
              "dummy potential, NOT the model. Only the 'md', 'pilot' and file columns tell you the plumbing works. ***\n")
    cols = ["system", "atoms", "build", "equiv", "md", "steps_per_s", "ns_per_day", "T_end_K", "Pt_detached_max",
            "backbone_broken_max", "SO3H_dissoc_max", "S_Pt_contact_frac", "Os_on_Pt_mean", "Pt_CN_mean", "pilot", "note"]
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
    print("Flags to read: backbone_broken_max > 0 = Nafion bonds other than O-H broke (unphysical); SO3H_dissoc_max = sulfonic-acid "
          "protons transferred to water (expected, compare with the number of sulfonates); Pt_detached_max > 0 = a Pt left the particle "
          "during plain MD (should not happen at 353 K without bias); S_Pt_contact_frac / Os_on_Pt_mean = how often sulfonate sits on Pt "
          "(second half of the run); steps_per_s sets the production budget.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()