#!/usr/bin/env python
"""Cross-check a LAMMPS ML-IAP run against the same MACE model in ASE.

usage: python check_vs_ase.py data.lammps your_original.model test_step0.lammpstrj [lammps_pe_eV]
Needs the ORIGINAL .model (not the exported *.pt), and the dump written by in.test_mace.lmp.
"""
import sys
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

data, model, dump = sys.argv[1:4]
Z = {1: 78, 2: 6, 3: 8, 4: 1, 5: 9, 6: 16}                    # Pt C O H F S, same as the data file

atoms = read(data, format="lammps-data", atom_style="atomic", Z_of_type=Z)
atoms.pbc = True
atoms.calc = MACECalculator(model_paths=model, device="cpu", default_dtype="float64")  # add head="..." if multihead
e_ase, f_ase = atoms.get_potential_energy(), atoms.get_forces()
print(f"ASE:    E = {e_ase:.4f} eV   max|F| = {np.abs(f_ase).max():.3f} eV/A")

lam = read(dump, format="lammps-dump-text")
f_lmp = lam.get_forces()
d = f_lmp - f_ase
print(f"LAMMPS vs ASE forces: RMSE = {np.sqrt((d**2).mean())*1000:.2f} meV/A   "
      f"max|dF| = {np.abs(d).max()*1000:.1f} meV/A")
if len(sys.argv) > 4:
    print(f"energy difference (LAMMPS - ASE) = {float(sys.argv[4]) - e_ase:+.4f} eV")

# same contact definitions as in.test_mace.lmp
sym = np.array(atoms.get_chemical_symbols())
npt = (sym == "Pt").sum()
for other, rc in [("Pt", 3.3), ("C", 2.6), ("O", 2.7), ("S", 3.6)]:
    i, j = neighbor_list("ij", atoms, rc)
    n = ((sym[i] == "Pt") & (sym[j] == other)).sum()
    label = f"mean Pt-Pt CN = {n/npt:.3f}" if other == "Pt" else f"total Pt-{other} contacts = {n}"
    print(f"  {label}  (cutoff {rc} A)")