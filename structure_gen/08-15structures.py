#!/usr/bin/env python
"""Pt nanoparticle / graphene / water starting structures for several NP sizes."""
import subprocess
import numpy as np
from ase import Atoms
from ase.build import graphene, make_supercell, molecule
from ase.cluster import wulff_construction
from ase.io import read, write
from ase.neighborlist import neighbor_list

A_PT, V_PT, RHO_W = 3.92, 3.92**3 / 4, 0.0334      # Å, Å^3/atom, molecules/Å^3
SURFACES = [(1, 1, 1), (1, 0, 0), (1, 1, 0)]
ENERGIES = [1.49, 1.73, 1.76]   # approx. relative Pt surface energies; only sets shape, tune freely

DIAMETERS_NM = [0.8, 1.0, 1.15, 1.3, 1.5]
MARGIN = 7.0        # graphene beyond NP edge (Å); NP-image gap = 2*MARGIN
Z_G = 5.0           # graphene height (keep off the cell boundary)
D_PT_C = 3.3        # initial NP-graphene gap; the MLIP relaxation will adjust it
WATER_ABOVE = 12.0  # water above NP top (Å)
VAC = 15.0          # vacuum above the water (Å)
TOL = 2.0
PACK_WATER = True


def n_from_diameter(d_nm):
    d = 10 * d_nm
    return max(4, round(np.pi / 6 * d**3 / V_PT))


def build_np(n_target):
    cl = wulff_construction("Pt", SURFACES, ENERGIES, n_target, "fcc",
                            rounding="closest", latticeconstant=A_PT)
    pt = Atoms(cl)
    pt.rotate((1, 1, 1), (0, 0, -1))        # (111) facet faces the support
    pt.center()
    return pt


def extent(a):
    p = a.positions
    return max(np.linalg.norm(p - q, axis=1).max() for q in p) + A_PT / np.sqrt(2)


def n_surface(pt):
    i, _ = neighbor_list("ij", pt, 3.3)
    return int((np.bincount(i, minlength=len(pt)) < 12).sum())


def build_graphene(Lx_min, Ly_min):
    gr = graphene(size=(1, 1, 1))
    gr = make_supercell(gr, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])   # rectangular cell
    nx = int(np.ceil(Lx_min / gr.cell[0, 0]))
    ny = int(np.ceil(Ly_min / gr.cell[1, 1]))
    return gr.repeat((nx, ny, 1))


def build_solid(n_pt):
    pt = build_np(n_pt)
    w = np.ptp(pt.positions[:, :2], axis=0)
    gr = build_graphene(w[0] + 2 * MARGIN, w[1] + 2 * MARGIN)
    Lx, Ly = gr.cell[0, 0], gr.cell[1, 1]
    gr.positions[:, 2] = Z_G
    p = pt.positions
    p[:, 0] += Lx / 2 - 0.5 * (p[:, 0].min() + p[:, 0].max())
    p[:, 1] += Ly / 2 - 0.5 * (p[:, 1].min() + p[:, 1].max())
    p[:, 2] += Z_G + D_PT_C - p[:, 2].min()
    return gr + pt, pt, Lx, Ly


def pack_water(solid, pt, Lx, Ly, tag):
    z_floor = Z_G + 2.5
    z_top = pt.positions[:, 2].max() + WATER_ABOVE
    v_excl = np.pi / 6 * (extent(pt) + TOL) ** 3
    n_w = int(0.92 * RHO_W * (Lx * Ly * (z_top - z_floor) - v_excl))
    write(f"solid_{tag}.xyz", solid)
    write("water.xyz", molecule("H2O"))
    open(f"pack_{tag}.inp", "w").write(f"""tolerance {TOL}
filetype xyz
seed 1234567
output liquid_{tag}.xyz
structure solid_{tag}.xyz
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure
structure water.xyz
  number {n_w}
  inside box {TOL/2} {TOL/2} {z_floor} {Lx - TOL/2} {Ly - TOL/2} {z_top}
end structure
""")
    subprocess.run(f"packmol < pack_{tag}.inp > pack_{tag}.log", shell=True, check=True)
    system = read(f"liquid_{tag}.xyz")
    system.set_cell(np.diag([Lx, Ly, z_top + VAC]))
    system.set_pbc(True)
    return system


def check(system, n_solid, tag):
    i, j, d = neighbor_list("ijd", system, 1.5)
    mol = lambda k: np.where(k >= n_solid, (k - n_solid) // 3, -1 - k)   # O,H,H per water
    bad = ~((i >= n_solid) & (j >= n_solid) & (mol(i) == mol(j)))
    dmin = d[bad].min() if bad.any() else None
    print(f"   non-bonded pairs <1.5 Å: {bad.sum()}  min: {dmin}")


if __name__ == "__main__":
    print(f"{'d_vol/nm':>8} {'N_Pt':>5} {'surf':>5} {'ext/nm':>7} {'Lx':>6} {'Ly':>6} {'N_atoms':>8}")
    for dn in DIAMETERS_NM:
        n_t = n_from_diameter(dn)
        solid, pt, Lx, Ly = build_solid(n_t)
        tag = f"d{int(round(dn*100)):03d}"
        system = solid
        if PACK_WATER:
            system = pack_water(solid, pt, Lx, Ly, tag)
            check(system, len(solid), tag)
        else:
            solid.set_cell(np.diag([Lx, Ly, solid.positions[:, 2].max() + VAC])); solid.set_pbc(True)
        write(f"data_{tag}.lammps", system, format="lammps-data", atom_style="atomic",
              specorder=["Pt", "C", "O", "H", "F", "S"], masses=True)
        dv = (28.76 * len(pt)) ** (1 / 3) / 10
        print(f"{dv:8.2f} {len(pt):5d} {n_surface(pt):5d} {extent(pt)/10:7.2f} "
              f"{Lx:6.1f} {Ly:6.1f} {len(system):8d}")