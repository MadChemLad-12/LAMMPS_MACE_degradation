import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from mace.calculators import MACECalculator

DATA  = "/home/user/Documents/Programs/For_GIT/LAMMPS_MACE_degradation/structure_gen/structures/octahedron_019pt/data.lammps"
MODEL = "/home/user/Documents/Programs/For_GIT/LAMMPS_MACE_degradation/mace_V5_active_learning_stagetwo_compiled.model"          # the .model file, NOT *-lammps.pt

atoms = read(DATA, format="lammps-data", atom_style="atomic",
             Z_of_type={1: 78, 2: 6, 3: 8, 4: 1, 5: 9, 6: 16})   # Pt C O H F S
atoms.pbc = True
atoms.calc = MACECalculator(model_paths=MODEL, device="cuda",
                            default_dtype="float32")            # add head="..." if multihead
print(f"number of atoms: {len(atoms)}")
print(f"Using model: {MODEL}")

f = atoms.get_forces()
print(f"{len(atoms)} atoms  E = {atoms.get_potential_energy():.3f} eV  "
      f"max|F| = {np.abs(f).max():.2f} eV/A")

FIRE(atoms, logfile="-").run(fmax=0.2, steps=300)               # relax packing contacts

MaxwellBoltzmannDistribution(atoms, temperature_K=100)
dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=353, friction=0.01 / units.fs)
traj = []
def log():
    f = np.abs(atoms.get_forces()).max()
    print(f"step {dyn.nsteps:4d}  T = {atoms.get_temperature():6.1f} K  "
          f"Epot = {atoms.get_potential_energy():.2f} eV  max|F| = {f:.2f}")
    traj.append(atoms.copy())
dyn.attach(log, interval=20)
dyn.run(400)                                                     # 200 fs
write("model_test.xyz", traj)