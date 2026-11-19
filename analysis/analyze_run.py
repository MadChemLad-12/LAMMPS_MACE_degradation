#!/usr/bin/env python3
"""
analyze_run.py - look at what a LAMMPS(+PLUMED) run of the Pt/graphene/Nafion/water system did.

Sub-commands (each writes PNG/CSV files next to --out):
  log      thermo + speed from a LAMMPS log:        analyze_run.py log run.log --nprod 4000000
  traj     structure metrics per frame from the extxyz written by ptgr_naf.lmp
           (Pt coordination/detachment, Pt-graphene anchoring, Pt-O and sulfonate contacts,
            Nafion bond integrity, water z-density):   analyze_run.py traj run.extxyz --out run
  colvar   PLUMED COLVAR time series, detachment events, SIGMA advice (unbiased runs):
                                                       analyze_run.py colvar COLVAR --out run
  fes      free energy from PLUMED HILLS (well-tempered): 2D map, 1D projection, and the
           attached-vs-detached free-energy difference vs simulated time = CONVERGENCE check:
                                                       analyze_run.py fes HILLS.0 HILLS.1 --bias-factor 15 --temp 353
  convert  fallback if your LAMMPS has no `dump extxyz`: lammpstrj -> extxyz (with forces and Timestep):
                                                       analyze_run.py convert run.lammpstrj --out run.extxyz
  merge    attach COLVAR values + energy/forces to the extxyz frames (for picking frames by CV
           value and for extract_frames.py --where):  analyze_run.py merge run.extxyz COLVAR --out run_cv.extxyz

Timing: PLUMED time = MD step x dt, and ptgr_naf.lmp restarts the step counter after the
minimisation, so step and time line up between log, extxyz, energy file and COLVAR.
"""
import argparse
import csv
import re
import sys

import numpy as np

KB = 8.617333e-5  # eV/K


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------------------------------
# log
# ------------------------------------------------------------------------------------------
def parse_log(path):
    blocks, perf, cur = [], [], None
    for line in open(path, errors="replace"):
        if cur is None:
            p = line.split()
            if len(p) >= 3 and p[0] == "Step" and "PotEng" in p:
                cur = dict(cols=p, rows=[])
            m = re.match(r"\s*Performance:\s+([\d.]+) ns/day,\s+[\d.]+ hours/ns,\s+([\d.]+) timesteps/s", line)
            if m:
                perf.append((float(m.group(1)), float(m.group(2))))
            continue
        if line.startswith("Loop time") or line.startswith("ERROR"):
            blocks.append(cur)
            cur = None
            continue
        p = line.split()
        if len(p) == len(cur["cols"]):
            try:
                cur["rows"].append([float(x) for x in p])
            except ValueError:
                pass
    if cur is not None and cur["rows"]:
        blocks.append(cur)
    m2 = re.findall(r"Performance:\s+([\d.]+) ns/day,\s+[\d.]+ hours/ns,\s+([\d.]+) timesteps/s",
                    open(path, errors="replace").read())
    return blocks, [(float(a), float(b)) for a, b in m2]


def cmd_log(a):
    blocks, perf = parse_log(a.log)
    if not blocks:
        raise SystemExit("no thermo blocks found (run LAMMPS with -log file, thermo header 'Step Time Temp PotEng ...')")
    print(f"{len(blocks)} thermo blocks; speed lines: {len(perf)}")
    for i, (nsd, sps) in enumerate(perf):
        print(f"  run {i + 1}: {sps:8.2f} steps/s = {nsd:7.3f} ns/day = {24 / max(nsd, 1e-9):7.1f} h/ns")
    if perf:
        sps = perf[-1][1]
        print(f"\nlast run speed: {sps:.2f} steps/s -> 1 ns = {a.dt_fs and 1e6 / a.dt_fs / sps / 3600:.1f} h at dt={a.dt_fs} fs")
        if a.nprod:
            h = a.nprod / sps / 3600
            print(f"{a.nprod} production steps = {a.nprod * a.dt_fs * 1e-6:.2f} ns -> {h:.1f} h ({h / 24:.1f} days) per walker/run")
    plt = plt_()
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    off = 0
    for b in blocks:
        c = {n: k for k, n in enumerate(b["cols"])}
        r = np.array(b["rows"])
        if not len(r):
            continue
        x = r[:, c["Step"]] + off
        ax[0].plot(x, r[:, c["Temp"]])
        ax[1].plot(x, r[:, c["PotEng"]])
        off = x[-1]
    ax[0].set(ylabel="T (K)")
    ax[1].set(ylabel="potential energy (eV)", xlabel="cumulative step (blocks joined end to end)")
    fig.tight_layout()
    fig.savefig(a.out + "_thermo.png", dpi=140)
    last = np.array(blocks[-1]["rows"])
    c = {n: k for k, n in enumerate(blocks[-1]["cols"])}
    h = len(last) // 2
    if h:
        print(f"last block, second half: T = {last[h:, c['Temp']].mean():.1f} +- {last[h:, c['Temp']].std():.1f} K, "
              f"PE drift over the block = {last[-1, c['PotEng']] - last[0, c['PotEng']]:.3f} eV")
    print(f"wrote {a.out}_thermo.png")
    print("NVT: total energy is NOT conserved (thermostat). Judge stability by T staying at the target, PE "
          "plateauing after equilibration and no spikes.")


# ------------------------------------------------------------------------------------------
# traj
# ------------------------------------------------------------------------------------------
def groups_from_first(at, n_graphene):
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    sym = np.array(at.get_chemical_symbols())
    n = len(at)
    isPt = sym == "Pt"
    pt = np.where(isPt)[0]
    gr = np.arange(n_graphene) if n_graphene else np.where((sym == "C") & (np.arange(n) < pt.min()))[0]
    isgr = np.zeros(n, bool)
    isgr[gr] = True
    rad = covalent_radii[at.get_atomic_numbers()]
    i, j, d = neighbor_list("ijd", at, 3.0)
    bond = (d < 1.25 * (rad[i] + rad[j])) & ~isgr[i] & ~isgr[j] & ~isPt[i] & ~isPt[j]
    bi, bj, bd = i[bond], j[bond], d[bond]
    g = coo_matrix((np.ones(len(bi)), (bi, bj)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    nafion = np.zeros(n, bool)
    for L in np.unique(lab[np.isin(sym, ["F", "S"])]):
        nafion |= lab == L
    nafion &= ~isgr & ~isPt
    pairs = [(x, y, dd) for x, y, dd in zip(bi, bj, bd) if x < y and nafion[x] and nafion[y]]
    isS = sym == "S"
    Os = np.zeros(n, bool)
    m = isS[i] & (sym[j] == "O") & (d < 1.9)
    Os[j[m]] = True
    nH = np.bincount(i[(sym[i] == "O") & (sym[j] == "H") & (d < 1.25)], minlength=n)
    ptO = np.zeros(n, bool)
    ptO[i[(sym[i] == "O") & isPt[j] & (d < 2.5)]] = True
    Oads = (sym == "O") & ~Os & ptO & (nH <= 1)
    Ow = (sym == "O") & ~Os & ~Oads & ~nafion
    return dict(sym=sym, pt=pt, gr=gr, nafion=nafion, bonds=np.array([(x, y) for x, y, _ in pairs]),
                bond_r0=np.array([dd for _, _, dd in pairs]), isS=isS, Os=Os, Oads=Oads, Ow=Ow)


def frame_metrics(at, G, pt_cut=3.2):
    from ase.neighborlist import neighbor_list
    n = len(at)
    i, j, d = neighbor_list("ijd", at, 3.2)
    sym = G["sym"]
    isPt = sym == "Pt"
    cn = np.bincount(i[isPt[i] & isPt[j] & (d < pt_cut)], minlength=n)[G["pt"]]
    i2, j2, d2 = neighbor_list("ijd", at, 3.6)
    m = np.zeros(n, bool)

    def minpt(mask):
        sel = isPt[i2] & mask[j2]
        out = np.full(n, 9.9)
        np.minimum.at(out, i2[sel], d2[sel])
        return out[G["pt"]]

    grm = np.zeros(n, bool)
    grm[G["gr"]] = True
    dC = minpt(grm)
    dS = minpt(G["isS"])

    def contacts(mask, cut=2.7):
        sel = isPt[i2] & mask[j2] & (d2 < cut)
        return int(len(np.unique(j2[sel])))

    all_O = sym == "O"
    cnO = np.bincount(i2[isPt[i2] & all_O[j2] & (d2 < 2.7)], minlength=n)[G["pt"]]
    brk = 0
    if len(G["bonds"]):
        dd = at.get_distances(G["bonds"][:, 0], G["bonds"][:, 1], mic=True)
        brk = int((dd > 1.3 * G["bond_r0"] + 0.25).sum())
    p = at.positions[G["pt"]]
    return dict(pt_cn_mean=cn.mean(), pt_cn_min=int(cn.min()), pt_detached=int((cn == 0).sum()),
                pt_rg=float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())),
                pt_c_contacts=int((dC < 2.8).sum()), pt_c_min=float(dC.min()), pt_s_min=float(dS.min()),
                n_Os_on_Pt=contacts(G["Os"]), n_Oads_on_Pt=contacts(G["Oads"]), n_Ow_on_Pt=contacts(G["Ow"]),
                pt_O_cn_mean=float(cnO.mean()), nafion_broken=brk)


def cmd_traj(a):
    from ase.io import iread
    first, rows, zO, zgr = None, [], [], None
    z_edges = np.arange(0, 80, 0.25)
    nz = 0
    for k, at in enumerate(iread(a.traj, index=f"{a.start}::{a.stride}")):
        if first is None:
            first = at
            G = groups_from_first(at, a.n_graphene)
            print(f"{len(at)} atoms | Pt {len(G['pt'])} | graphene C {len(G['gr'])} | Nafion atoms "
                  f"{int(G['nafion'].sum())} ({len(G['bonds'])} bonds tracked) | sulfonate O {int(G['Os'].sum())}, "
                  f"adsorbed O {int(G['Oads'].sum())}, water O {int(G['Ow'].sum())}")
            zgr = at.positions[G["gr"], 2].max()
        step = at.info.get("Timestep", a.start + k * a.stride)
        m = frame_metrics(at, G)
        m.update(step=int(step), time_ps=float(step) * a.dt, E_eV=float(at.info.get("Potential_energy", np.nan)))
        rows.append(m)
        h, _ = np.histogram(at.positions[G["Ow"], 2], bins=z_edges)
        zO.append(h)
        nz += 1
    if not rows:
        raise SystemExit("no frames read")
    keys = list(rows[0])
    with open(a.out + "_traj_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    t = np.array([r["time_ps"] for r in rows])
    arr = {k: np.array([r[k] for r in rows], float) for k in keys}
    h2 = len(t) // 2
    print(f"\n{len(t)} frames, {t[0]:.2f}-{t[-1]:.2f} ps  (mean +- sd over the second half)")
    for k in keys[:-3] + ["E_eV"]:
        if k in ("step", "time_ps"):
            continue
        v = arr[k][h2:]
        print(f"  {k:<16}{v.mean():>10.3f} +- {v.std():.3f}   [first {arr[k][0]:.3f}, last {arr[k][-1]:.3f}]")
    flags = []
    if arr["pt_detached"].max() > 0:
        k0 = int(np.argmax(arr["pt_detached"] > 0))
        flags.append(f"Pt atom(s) with no Pt neighbour from t = {t[k0]:.2f} ps (max {int(arr['pt_detached'].max())})")
    if arr["nafion_broken"].max() > 0:
        flags.append(f"Nafion bonds broken in some frames (max {int(arr['nafion_broken'].max())}) - unphysical unless intended")
    print("FLAGS: " + ("; ".join(flags) if flags else "none"))
    plt = plt_()
    panels = [("pt_cn_mean", "mean Pt-Pt CN"), ("pt_detached", "Pt with CN=0"), ("pt_c_contacts", "Pt within 2.8 A of C"),
              ("pt_s_min", "closest S to Pt (A)"), ("n_Os_on_Pt", "sulfonate O on Pt"), ("n_Ow_on_Pt", "water O on Pt"),
              ("n_Oads_on_Pt", "adsorbed O/OH on Pt"), ("nafion_broken", "Nafion bonds broken"), ("E_eV", "potential energy (eV)")]
    fig, ax = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
    for x, (k, lab) in zip(ax.ravel(), panels):
        x.plot(t, arr[k], lw=1)
        x.set_ylabel(lab, fontsize=9)
    for x in ax[-1]:
        x.set_xlabel("time (ps)")
    fig.tight_layout()
    fig.savefig(a.out + "_traj_metrics.png", dpi=140)
    Z = np.sum(zO[h2:], axis=0)
    zc = 0.5 * (z_edges[1:] + z_edges[:-1])
    area = first.cell[0, 0] * first.cell[1, 1]
    # water density: molecules / (area*dz) -> g/cc (18.015 g/mol)
    dens = Z / max(len(zO[h2:]), 1) / (area * 0.25) * 18.015 / 6.022e23 * 1e24
    fig, x = plt.subplots(figsize=(6, 4))
    x.plot(zc - zgr, dens)
    x.axhline(0.97, color="gray", ls=":", lw=1)
    x.set(xlabel="z above top graphene layer (A)", ylabel="water density (g/cm3, cell-area normalised)",
          xlim=(-2, 40))
    fig.tight_layout()
    fig.savefig(a.out + "_zdensity.png", dpi=140)
    core = (zc - zgr > a.bulk_lo) & (zc - zgr < a.bulk_hi)
    print(f"water density in z = {a.bulk_lo:.0f}-{a.bulk_hi:.0f} A above graphene: {dens[core].mean():.2f} g/cm3 "
          f"(dotted line = 0.97, ~PBE-D3 water is NOT guaranteed to match; this is a sanity check, and the region "
          f"includes the Pt particle's volume, so it reads low near the particle)")
    print(f"wrote {a.out}_traj_metrics.csv/.png, {a.out}_zdensity.png")


# ------------------------------------------------------------------------------------------
# colvar
# ------------------------------------------------------------------------------------------
def load_plumed(path):
    names = None
    rows = []
    for line in open(path):
        if line.startswith("#! FIELDS"):
            names = line.split()[2:]
        elif not line.startswith("#") and line.strip():
            rows.append([float(x) for x in line.split()])
    if names is None:
        raise SystemExit(f"{path}: no '#! FIELDS' header")
    r = np.array(rows)
    return {n: r[:, k] for k, n in enumerate(names)}


def cmd_colvar(a):
    d = load_plumed(a.colvar)
    t = d["time"]
    sk = t >= t[0] + a.skip_ps      # RELATIVE to the first COLVAR row: after a restart, PLUMED time continues from the MD step counter
    span = t[-1] - t[0]
    if sk.sum() < 5:
        print(f"note: --skip-ps {a.skip_ps} would leave {int(sk.sum())} of {len(t)} rows (this file spans only {span:.2f} ps); using all rows")
        sk = np.ones(len(t), bool)
    if span < 10.0:
        print(f"CAUTION: this COLVAR spans only {span:.2f} ps - means and fluctuations (and any SIGMA advice) are not reliable; "
              "use a pilot of at least 20-50 ps (40,000-100,000 steps at 0.5 fs)")
    cv = [k for k in d if k != "time"]
    print(f"{len(t)} rows, {t[0]:.2f}-{t[-1]:.2f} ps; statistics use {int(sk.sum())} rows (from t = {t[sk][0]:.2f} ps)")
    print(f"{'CV':<14}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}")
    for k in cv:
        v = d[k][sk]
        print(f"{k:<14}{v.mean():>9.3f}{v.std():>8.3f}{v.min():>8.3f}{v.max():>8.3f}")
    if "cn_ptpt" in d:
        det = d["cn_ptpt"] < a.detach_cn
        if det.any():
            print(f"\ncn_ptpt < {a.detach_cn} first at t = {t[np.argmax(det)]:.2f} ps; fraction of time detached "
                  f"{det.mean():.3f}")
        else:
            print(f"\ncn_ptpt never below {a.detach_cn}: no detachment in this run")
    biased = "metad.bias" in d
    if not biased and "cn_ptpt" in d and "cn_pto" in d:
        s1, s2 = d["cn_ptpt"][sk].std(), d["cn_pto"][sk].std()
        g1, g2 = max(0.5 * s1, 0.1), max(0.5 * s2, 0.1)
        print(f"\nunbiased fluctuation sd = {s1:.2f}, {s2:.2f} -> suggested SIGMA about {g1:.2f},{g2:.2f} "
              f"(half the sd, floored at 0.1). prep_plumed.py --sigma {g1:.2f},{g2:.2f}")
    elif biased:
        print("\nbiased run: sd values above are NOT unbiased fluctuations - use an unbiased run for SIGMA.")
    plt = plt_()
    keys = [k for k in ("cn_ptpt", "cn_pto", "cn_ptoa", "cn_ptow", "cn_ptos", "cn_pts", "dz.z", "dS.min", "dC.min",
                        "metad.bias") if k in d]
    fig, ax = plt.subplots(len(keys), 1, figsize=(9, 1.6 * len(keys) + 1), sharex=True)
    ax = np.atleast_1d(ax)
    for x, k in zip(ax, keys):
        x.plot(t, d[k], lw=0.8)
        x.set_ylabel(k, fontsize=8)
    ax[-1].set_xlabel("time (ps)")
    fig.tight_layout()
    fig.savefig(a.out + "_colvar.png", dpi=140)
    if "cn_ptpt" in d and "cn_pto" in d:
        fig, x = plt.subplots(figsize=(5, 4))
        h = x.hist2d(d["cn_ptpt"][sk], d["cn_pto"][sk], bins=50, cmap="viridis_r", cmin=1)
        x.set(xlabel="cn_ptpt (Pt-Pt CN of target)", ylabel="cn_pto (Pt-O CN of target)")
        fig.colorbar(h[3], label="counts" + (" (biased - not a free energy)" if biased else ""))
        fig.tight_layout()
        fig.savefig(a.out + "_colvar2d.png", dpi=140)
    print(f"wrote {a.out}_colvar.png, {a.out}_colvar2d.png")


# ------------------------------------------------------------------------------------------
# fes (well-tempered metadynamics, sum of Gaussians)
# ------------------------------------------------------------------------------------------
def load_hills(paths):
    H = []
    for p in paths:
        d = load_plumed(p)
        names = list(d)
        cv = [n for n in names if n not in ("time", "height", "biasf") and not n.startswith("sigma_")]
        H.append(np.column_stack([d["time"]] + [d[c] for c in cv] + [d["sigma_" + c] for c in cv] + [d["height"]]))
    H = np.vstack(H)
    return H[np.argsort(H[:, 0])], cv


def bias_grid(H, gx, gy):
    t, x, y, sx, sy, h = H.T
    Gx = np.exp(-0.5 * ((gx[:, None] - x[None]) / sx[None]) ** 2) * h[None]
    Gy = np.exp(-0.5 * ((gy[:, None] - y[None]) / sy[None]) ** 2)
    return Gx @ Gy.T


def cmd_fes(a):
    H, cv = load_hills(a.hills)
    if len(cv) != 2:
        raise SystemExit(f"expected 2 CVs in HILLS, found {cv}")
    kT = KB * a.temp
    gamma = a.bias_factor
    gx = np.arange(a.xmin, a.xmax, a.step)
    gy = np.arange(a.ymin, a.ymax, a.step)
    T = H[-1, 0]
    nsl = a.slices
    edges = np.linspace(0, T, nsl + 1)[1:]
    print(f"{len(H)} hills from {len(a.hills)} file(s), {H[0, 0]:.2f}-{T:.2f} ps, CVs {cv}, kT = {kT:.4f} eV, gamma = {gamma}")
    V = np.zeros((len(gx), len(gy)))
    res, k0 = [], 0
    fes_t = []
    for e in edges:
        sel = H[:, 0] <= e + 1e-9
        k1 = sel.sum()
        if k1 > k0:
            V += bias_grid(H[k0:k1], gx, gy)
            k0 = k1
        F = -gamma / (gamma - 1.0) * V
        F -= F.min()
        fes_t.append(F.copy())
        w = np.exp(-F / kT)
        att = (gx[:, None] > a.attached_cn) & np.ones_like(F, bool)
        det = (gx[:, None] < a.detached_cn) & np.ones_like(F, bool)
        Fa = -kT * np.log(np.sum(w[att[:, 0], :]))
        Fd = -kT * np.log(np.sum(w[det[:, 0], :]))
        F1 = -kT * np.log(np.sum(w, axis=1))
        F1 -= F1.min()
        ia = np.where(gx > a.attached_cn)[0]
        idt = np.where(gx < a.detached_cn)[0]
        # barrier: highest point of F1 between the attached minimum and the detached region
        ka = ia[np.argmin(F1[ia])] if len(ia) else None
        kd = idt[np.argmin(F1[idt])] if len(idt) else None
        bar = F1[min(ka, kd):max(ka, kd) + 1].max() - F1[ka] if ka is not None and kd is not None else np.nan
        res.append((e, Fd - Fa, bar))
    print(f"\n{'t (ps)':>9}{'dF detached-attached (eV)':>28}{'barrier on cn_ptpt (eV)':>26}")
    for e, dF, bar in res:
        print(f"{e:>9.1f}{dF:>28.3f}{bar:>26.3f}")
    last = np.array([r[1] for r in res[-3:]])
    print(f"\nconvergence: dF over the last 3 slices spans {last.max() - last.min():.3f} eV; the 1D barrier over the last "
          f"3 slices spans {np.ptp([r[2] for r in res[-3:]]):.3f} eV. If these still drift by more than ~0.1 eV, "
          "the run is not converged - do not quote a barrier.")
    print(f"basins: attached = {cv[0]} > {a.attached_cn}, detached = {cv[0]} < {a.detached_cn} (change with --attached-cn/--detached-cn)")
    plt = plt_()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    F = fes_t[-1]
    im = ax[0].contourf(gx, gy, np.clip(F, 0, a.fmax).T, 30, cmap="viridis_r")
    fig.colorbar(im, ax=ax[0], label="F (eV)")
    ax[0].set(xlabel=cv[0], ylabel=cv[1], title=f"FES at {T:.0f} ps (well-tempered, gamma={gamma:g})")
    for k, Fk in enumerate(fes_t):
        w = np.exp(-Fk / kT)
        F1 = -kT * np.log(w.sum(axis=1))
        ax[1].plot(gx, F1 - F1.min(), color=plt.cm.viridis(k / max(len(fes_t) - 1, 1)), lw=1)
    ax[1].set(xlabel=cv[0], ylabel="F(1D projection) (eV)", title="1D projection, early -> late (colour)")
    ax[2].plot([r[0] for r in res], [r[1] for r in res], "o-")
    ax[2].set(xlabel="simulated time (ps)", ylabel="dF detached - attached (eV)", title="convergence")
    fig.tight_layout()
    fig.savefig(a.out + "_fes.png", dpi=140)
    np.savez(a.out + "_fes.npz", gx=gx, gy=gy, F=F, cv=np.array(cv))
    print(f"wrote {a.out}_fes.png and {a.out}_fes.npz")
    print("note: the FES is only as good as the model (check committee/DFT), and the CVs are generic; "
          "mechanism comes from analysing frames in the saddle region (see 'merge').")


# ------------------------------------------------------------------------------------------
# merge
# ------------------------------------------------------------------------------------------
def cmd_merge(a):
    from ase.io import iread, write
    d = load_plumed(a.colvar)
    ts = d["time"]
    keys = [k for k in d if k != "time"]
    out, miss = [], 0
    for at in iread(a.traj, index=":"):
        step = at.info.get("Timestep", 0)
        t = float(step) * a.dt
        k = int(np.argmin(np.abs(ts - t)))
        if abs(ts[k] - t) > 0.6 * a.dt * a.colvar_stride:
            miss += 1
        else:
            for key in keys:
                at.info[key.replace(".", "_")] = float(d[key][k])
        at.info["md_step"], at.info["time_ps"] = int(step), t
        e = at.info.get("Potential_energy")
        if e is not None:
            at.info["energy"] = float(e)      # extxyz readers turn info['energy'] + arrays['forces'] into labels
        out.append(at)
    write(a.out, out, format="extxyz")
    print(f"wrote {len(out)} frames to {a.out} ({miss} frames without a COLVAR row within tolerance); "
          f"CV keys: {[k.replace('.', '_') for k in keys]}")


def cmd_convert(a):
    from ase.io import iread, write
    steps = []
    with open(a.traj) as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                steps.append(int(next(fh)))
    frames = []
    for k, at in enumerate(iread(a.traj, index=":", format="lammps-dump-text", specorder=a.specorder.split(","))):
        at.info["Timestep"] = steps[k]
        if at.calc is not None:
            try:
                at.arrays["forces"] = at.get_forces()
            except Exception:
                pass
        at.calc = None
        frames.append(at)
    write(a.out, frames, format="extxyz")
    print(f"wrote {len(frames)} frames to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("log")
    p.add_argument("log")
    p.add_argument("--nprod", type=int, default=0)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--out", default="run")
    p.set_defaults(f=cmd_log)
    p = sub.add_parser("traj")
    p.add_argument("traj")
    p.add_argument("--out", default="run")
    p.add_argument("--dt", type=float, default=0.0005, help="ps per MD step")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-graphene", type=int, default=None)
    p.add_argument("--bulk-lo", type=float, default=8.0, help="water density window above graphene (A)")
    p.add_argument("--bulk-hi", type=float, default=20.0)
    p.set_defaults(f=cmd_traj)
    p = sub.add_parser("colvar")
    p.add_argument("colvar")
    p.add_argument("--out", default="run")
    p.add_argument("--skip-ps", type=float, default=0.0, help="ps to skip at the START of this COLVAR file (relative)")
    p.add_argument("--detach-cn", type=float, default=0.5)
    p.set_defaults(f=cmd_colvar)
    p = sub.add_parser("fes")
    p.add_argument("hills", nargs="+")
    p.add_argument("--bias-factor", type=float, required=True)
    p.add_argument("--temp", type=float, default=353.0)
    p.add_argument("--out", default="run")
    p.add_argument("--xmin", type=float, default=-1.0)
    p.add_argument("--xmax", type=float, default=12.0)
    p.add_argument("--ymin", type=float, default=-1.0)
    p.add_argument("--ymax", type=float, default=10.0)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--slices", type=int, default=8)
    p.add_argument("--attached-cn", type=float, default=3.0)
    p.add_argument("--detached-cn", type=float, default=1.0)
    p.add_argument("--fmax", type=float, default=3.0)
    p.set_defaults(f=cmd_fes)
    p = sub.add_parser("convert")
    p.add_argument("traj")
    p.add_argument("--out", default="converted.extxyz")
    p.add_argument("--specorder", default="Pt,C,O,H,F,S")
    p.set_defaults(f=cmd_convert)
    p = sub.add_parser("merge")
    p.add_argument("traj")
    p.add_argument("colvar")
    p.add_argument("--out", default="merged.extxyz")
    p.add_argument("--dt", type=float, default=0.0005)
    p.add_argument("--colvar-stride", type=int, default=100)
    p.set_defaults(f=cmd_merge)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()