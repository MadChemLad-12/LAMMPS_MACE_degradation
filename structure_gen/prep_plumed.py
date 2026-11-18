#!/usr/bin/env python3
"""
prep_plumed.py - turn a LAMMPS data file (or an equilibrated snapshot) into a ready-to-run
PLUMED input for "dissolution of ONE chosen Pt atom", and list the candidate target atoms.

  python prep_plumed.py data.lammps --list-sites
  python prep_plumed.py equil.data --site vertex --mode unbiased --outdir run_vertex_unbiased
  python prep_plumed.py equil.data --site id:305 --mode metad --outdir run_305 --walkers 2 --walker-id 0

Run it on an EQUILIBRATED snapshot (write_data after the equilibration stage), not on the raw
Packmol structure: site classes use coordination numbers and are only meaningful after relaxation.

What is defined (all by atom index, fixed for the run)
  gr / gr_top    graphene C = C atoms that come before the first Pt in the file (builder order:
                 graphene C, Pt, Nafion, water); gr_top = those within 1 A of the highest layer.
                 --n-graphene N overrides. Nafion carbon is NOT part of gr.
  sulf           S atoms.   o_s = O bonded to S (<1.9 A).
  o_ads          O (or OH) bound to Pt in the initial structure: Pt neighbour < 2.5 A and <= 1 H.
  o_w            all other O (water).  Roles are fixed by index; check they stay valid if O/H exchange.
Biased CVs (metad mode): cn_ptpt (target Pt-Pt coordination), cn_pto (target-O coordination, all O).
Monitored only: cn_ptoa / cn_ptow / cn_ptos (Pt-O by O type), dz.z (height above top graphene
layer), dS.min (closest S), cn_pts, dC.min (closest graphene C, anchoring).
Site classes (from Pt-Pt CN within 3.2 A; fcc nanoparticle rules of thumb):
  perimeter = a graphene C within --c-cut (2.8 A) of the Pt (support contact)
  vertex CN<=5, edge 6-7, facet 8-9, interior >=10.

--mode unbiased writes the same monitored CVs without METAD (use it to measure CV
fluctuations for SIGMA and to check the CVs behave) ; --mode metad adds the bias.
"""
import argparse
import os

import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list


def ranges(ids):
    """[1,2,3,7] (1-based) -> '1-3,7'"""
    ids = sorted(int(i) for i in ids)
    if not ids:
        raise SystemExit("empty atom group")
    out, s, p = [], ids[0], ids[0]
    for i in ids[1:] + [None]:
        if i is not None and i == p + 1:
            p = i
            continue
        out.append(f"{s}-{p}" if p > s else f"{s}")
        s = p = i
    return ",".join(out)


def classify(cn, perim):
    if perim:
        return "perimeter"
    if cn <= 5:
        return "vertex"
    if cn <= 7:
        return "edge"
    if cn <= 9:
        return "facet"
    return "interior"


def analyse(args):
    spec = args.specorder.split(",")
    Z = {i + 1: atomic_numbers[e] for i, e in enumerate(spec)}
    at = read(args.data, format="lammps-data", atom_style="atomic", Z_of_type=Z)
    at.set_pbc((True, True, False) if args.nonperiodic_z else True)
    sym = np.array(at.get_chemical_symbols())
    n = len(at)
    isPt, isC, isO, isH, isS = (sym == s for s in ("Pt", "C", "O", "H", "S"))
    pt = np.where(isPt)[0]
    if not len(pt):
        raise SystemExit("no Pt in this file")
    if args.n_graphene is not None:
        gr = np.arange(args.n_graphene)
        if not isC[gr].all():
            raise SystemExit("--n-graphene: first N atoms are not all C")
    else:
        gr = np.where(isC & (np.arange(n) < pt.min()))[0]
    if not len(gr):
        raise SystemExit("could not identify graphene C (no C before the first Pt); use --n-graphene")
    ztop = at.positions[gr, 2].max()
    gr_top = gr[at.positions[gr, 2] > ztop - 1.0]

    i, j, d = neighbor_list("ijd", at, 3.6)
    Os = np.zeros(n, bool)
    sel = isS[i] & isO[j] & (d < 1.9)
    Os[j[sel]] = True
    nH = np.bincount(i[isO[i] & isH[j] & (d < 1.25)], minlength=n)
    bound_pt = np.zeros(n, bool)
    bound_pt[i[isO[i] & isPt[j] & (d < 2.5)]] = True
    Oads = isO & ~Os & bound_pt & (nH <= 1)
    Ow = isO & ~Os & ~Oads

    ptcn = np.bincount(i[isPt[i] & isPt[j] & (d < 3.2)], minlength=n)
    dC = np.full(n, 9.9)
    m = isPt[i] & np.isin(j, gr)
    np.minimum.at(dC, i[m], d[m])
    dS = np.full(n, 9.9)
    m = isPt[i] & isS[j]
    np.minimum.at(dS, i[m], d[m])
    n_oads = np.bincount(i[isPt[i] & Oads[j] & (d < 2.6)], minlength=n)
    rows = []
    for k in pt:
        rows.append(dict(id=k + 1, cn=int(ptcn[k]), z=at.positions[k, 2] - ztop, dC=dC[k], dS=dS[k],
                         noads=int(n_oads[k]), cls=classify(ptcn[k], dC[k] < args.c_cut)))
    return at, dict(gr=gr, gr_top=gr_top, sulf=np.where(isS)[0], o_s=np.where(Os)[0],
                    o_ads=np.where(Oads)[0], o_w=np.where(Ow)[0], pt=pt, ztop=ztop,
                    ztop_liquid=at.positions[isO, 2].max() if isO.any() else ztop + 20, rows=rows)


def pick(rows, site):
    if site.startswith("id:"):
        k = int(site[3:])
        r = [r for r in rows if r["id"] == k]
        if not r:
            raise SystemExit(f"atom {k} is not a Pt atom")
        return r[0]
    c = [r for r in rows if r["cls"] == site]
    if not c:
        raise SystemExit(f"no Pt atom of class '{site}' (try --list-sites; classes: "
                         f"{sorted({r['cls'] for r in rows})})")
    # exposed ones first: highest z for vertex/edge/facet, closest to carbon for perimeter
    return min(c, key=lambda r: r["dC"]) if site == "perimeter" else max(c, key=lambda r: r["z"])


def write_plumed(args, G, tgt, path):
    t = tgt["id"]
    rest = [k + 1 for k in G["pt"] if k + 1 != t]
    zwall = args.zwall if args.zwall else tgt["z"] + args.zwall_pad
    zcap = G["ztop_liquid"] - G["ztop"] - 3.0
    if zwall > zcap:
        print(f"note: wall {zwall:.1f} A reduced to {zcap:.1f} A (top of the liquid minus 3 A)")
        zwall = zcap
    sig = args.sigma
    L = ["UNITS LENGTH=A ENERGY=eV            # LAMMPS fix converts units; this file is in A and eV",
         f"# target Pt atom {t} ({tgt['cls']}), Pt-Pt CN={tgt['cn']}, height above top graphene layer {tgt['z']:.2f} A",
         f"# generated by prep_plumed.py from {os.path.basename(args.data)}; mode={args.mode}", ""]
    if args.restart:
        L += ["RESTART", ""]
    L += [f"pt_rest: GROUP ATOMS={ranges(rest)}      # all Pt except the target",
          f"gr:      GROUP ATOMS={ranges(G['gr'] + 1)}      # graphene C only (not Nafion C)",
          f"gr_top:  GROUP ATOMS={ranges(G['gr_top'] + 1)}",
          f"oall:    GROUP ATOMS={ranges(np.concatenate([G['o_ads'], G['o_s'], G['o_w']]) + 1)}   # every O"]
    for name, key, txt in (("o_ads", "o_ads", "adsorbed O/OH at t=0"), ("o_s", "o_s", "sulfonate O"),
                           ("o_w", "o_w", "water O"), ("sulf", "sulf", "S")):
        if len(G[key]):
            L.append(f"{name:<8} GROUP ATOMS={ranges(G[key] + 1)}      # {txt}".replace(name, name + ":", 1))
    L += ["", "# ---- collective variables (target Pt = atom %d) ----" % t,
          f"cn_ptpt: COORDINATION GROUPA={t} GROUPB=pt_rest R_0=3.3 NN=6 MM=12",
          f"cn_pto:  COORDINATION GROUPA={t} GROUPB=oall    R_0=2.7 NN=6 MM=12"]
    mon = ["cn_ptpt", "cn_pto"]
    if len(G["o_ads"]):
        L.append(f"cn_ptoa: COORDINATION GROUPA={t} GROUPB=o_ads R_0=2.7 NN=6 MM=12   # monitored")
        mon.append("cn_ptoa")
    L.append(f"cn_ptow: COORDINATION GROUPA={t} GROUPB=o_w    R_0=2.7 NN=6 MM=12   # monitored")
    mon.append("cn_ptow")
    if len(G["sulf"]):
        L += [f"cn_ptos: COORDINATION GROUPA={t} GROUPB=o_s    R_0=2.7 NN=6 MM=12   # monitored",
              f"cn_pts:  COORDINATION GROUPA={t} GROUPB=sulf   R_0=3.6 NN=6 MM=12   # monitored",
              f"dS:      DISTANCES GROUPA={t} GROUPB=sulf MIN={{BETA=50}}            # monitored: closest S"]
        mon += ["cn_ptos", "cn_pts", "dS.min"]
    L += [f"dC:      DISTANCES GROUPA={t} GROUPB=gr MIN={{BETA=50}}              # monitored: closest graphene C",
          "cc:      CENTER ATOMS=gr_top",
          f"dz:      DISTANCE ATOMS=cc,{t} COMPONENTS NOPBC      # vector cc->target: dz.z = height ABOVE graphene",
          "", f"uwall: UPPER_WALLS ARG=dz.z AT={zwall:.2f} KAPPA=5.0      # keeps a detached ion in the simulated liquid"]
    mon += ["dC.min", "dz.z"]
    if args.mode == "metad":
        L += ["", "METAD ...", "       LABEL=metad",
              f"       ARG=cn_ptpt,cn_pto SIGMA={sig[0]},{sig[1]} HEIGHT={args.height} PACE={args.pace}",
              f"       BIASFACTOR={args.biasfactor} TEMP={args.temp} GRID_MIN=-1,-1 GRID_MAX=12,10",
              "       FILE=HILLS CALC_RCT"]
        if args.walkers > 1:
            L += [f"       WALKERS_N={args.walkers} WALKERS_ID={args.walker_id} "
                  f"WALKERS_DIR={args.walkers_dir} WALKERS_RSTRIDE={args.pace}"]
        L += ["... METAD"]
        mon += ["metad.bias", "metad.rbias"]
    L += ["", f"PRINT ARG={','.join(mon)} STRIDE={args.stride} FILE=COLVAR", ""]
    open(path, "w").write("\n".join(L))
    return zwall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--specorder", default="Pt,C,O,H,F,S", help="element of LAMMPS type 1,2,...")
    ap.add_argument("--nonperiodic-z", action="store_true", default=True)
    ap.add_argument("--n-graphene", type=int, default=None)
    ap.add_argument("--c-cut", type=float, default=2.8)
    ap.add_argument("--list-sites", action="store_true")
    ap.add_argument("--site", default="vertex", help="vertex|edge|facet|perimeter|interior|id:N")
    ap.add_argument("--mode", default="metad", choices=["metad", "unbiased"])
    ap.add_argument("--outdir", default="plumed_run")
    ap.add_argument("--sigma", default="0.3,0.3", type=lambda s: [float(x) for x in s.split(",")],
                    help="hill widths; set to ~half the CV fluctuation from an unbiased run")
    ap.add_argument("--height", type=float, default=0.02, help="hill height, eV")
    ap.add_argument("--pace", type=int, default=500, help="MD steps between hills")
    ap.add_argument("--biasfactor", type=float, default=15.0)
    ap.add_argument("--temp", type=float, default=353.0)
    ap.add_argument("--stride", type=int, default=100, help="COLVAR print stride (MD steps)")
    ap.add_argument("--zwall", type=float, default=None, help="wall on target height above top graphene (A)")
    ap.add_argument("--zwall-pad", type=float, default=6.0, help="default wall = initial height + pad")
    ap.add_argument("--walkers", type=int, default=1)
    ap.add_argument("--walker-id", type=int, default=0)
    ap.add_argument("--walkers-dir", default="../walkers")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    at, G = analyse(args)
    print(f"{len(at)} atoms | Pt {len(G['pt'])} | graphene C {len(G['gr'])} (top layer {len(G['gr_top'])}, "
          f"z={G['ztop']:.2f} A) | S {len(G['sulf'])} | O: sulfonate {len(G['o_s'])}, adsorbed {len(G['o_ads'])}, "
          f"water {len(G['o_w'])}")
    if args.list_sites:
        print(f"{'id':>5} {'class':<10}{'CN_Pt':>6}{'z_above_gr':>11}{'d(C)':>7}{'d(S)':>7}{'nO*':>5}")
        for r in sorted(G["rows"], key=lambda r: (r["cls"], -r["z"])):
            print(f"{r['id']:>5} {r['cls']:<10}{r['cn']:>6}{r['z']:>11.2f}{r['dC']:>7.2f}{r['dS']:>7.2f}{r['noads']:>5}")
        return
    tgt = pick(G["rows"], args.site)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "plumed.dat")
    zw = write_plumed(args, G, tgt, path)
    print(f"target: Pt {tgt['id']} ({tgt['cls']}, CN {tgt['cn']}, {tgt['z']:.1f} A above graphene, "
          f"nearest S {tgt['dS']:.1f} A) | wall at {zw:.1f} A | wrote {path}")


if __name__ == "__main__":
    main()
