import os
from ase.io import read, write
from ase.calculators.cp2k import CP2K
from ase.units import Rydberg
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 1. Load the MACE trajectory (index=":" reads all frames)
traj = read("selected.extxyz", index=":")

# 2. Load your base CP2K template
with open("template.inp", "r") as f:
    base_inp = f.read()

# 3. Configure the CP2K calculator
# Note: ASE runs a persistent CP2K shell process, avoiding startup overhead between frames!
calc = CP2K(
    command="cp2k.ssmp --shell",  # Use cp2k_shell for ASE
    inp=base_inp,
    basis_set=None,       # Defined in template.inp
    basis_set_file=None,  # Defined in template.inp
    potential_file=None,  # Defined in template.inp
    pseudo_potential=None,# Defined in template.inp
    xc=None,              # Defined in template.inp
    cutoff=None,          # Defined in template.inp
    charge=0,
    max_scf=200,
    stress_tensor=True,   # Set False if you don't need REF_stress
)

output_file = "cp2k_dft_ref.xyz"
if os.path.exists(output_file):
    os.remove(output_file)

for i, atoms in enumerate(traj):
    print(f"Running DFT single-point on frame {i+1}/{len(traj)}...")
    
    # Preserve original MACE predictions under custom info/array keys if desired
    if "energy" in atoms.info:
        atoms.info["MACE_energy"] = atoms.info.pop("energy")
    if "forces" in atoms.arrays:
        atoms.arrays["MACE_forces"] = atoms.arrays.pop("forces")

    # Attach calculator and compute DFT energy & forces
    atoms.calc = calc
    dft_energy = atoms.get_potential_energy()
    dft_forces = atoms.get_forces()
    
    # Store with MACE-compatible reference keys (REF_energy, REF_forces)
    atoms.info["REF_energy"] = dft_energy
    atoms.arrays["REF_forces"] = dft_forces
    if calc.results.get("stress") is not None:
        atoms.info["REF_stress"] = atoms.get_stress()

    # Clear calculator before writing so ASE writes custom REF_ keys cleanly
    atoms.calc = None
    write(output_file, atoms, append=True)

print("Batch calculation complete!")


