# LAMMPS-MACE Pt/Graphene/Nafion Degradation Pipeline

An automated molecular dynamics (MD) and machine-learning interatomic potential (MLIP) workflow for simulating Platinum (Pt) nanoparticles supported on graphene in an aqueous Nafion electrolyte environment using **MACE** (via LAMMPS ML-IAP/Kokkos) and optional **PLUMED** enhanced sampling.

---

## 📁 Repository Architecture

Based on your workspace layout, the repository is organized into four core functional areas:

```text
LAMMPS_MACE_degradation/
├── run_pipeline.sh               # Master orchestration script (build -> equiv -> MD -> analysis -> summary)
├── enviroment.yml                # Conda environment specification (lmp_mace)
├── mace_V9_active_learning_...   # Active learning MACE models (.model for ASE, -mliap_lammps.pt for LAMMPS)
├── mace_V5_active_learning_...   # Previous generation MACE models (tracked via DVC)
│
├── structure_gen/                # Atomic interface generation & Packmol wrappers
│   ├── sweep_build.py            # Generates parameter sweeps (sizes, coverages, seeds, Nafion modes)
│   ├── build_interface.py        # Builds Pt nanoparticle + graphene + Nafion + water slabs
│   ├── adsorbate_builder.py      # Places O/OH adsorbates on exposed Pt facets/sites
│   ├── prep_plumed.py            # Generates PLUMED collective variable (COLVAR) input files
│   └── structures_sweep/         # Output directory containing generated data.lammps & manifest.txt
│
├── lammps/                       # LAMMPS simulation templates and potential definitions
│   ├── ptgr_naf/
│   │   ├── ptgr_naf.lmp          # Main MD protocol (Minimize -> Heat -> Equilibrate -> Production)
│   │   ├── pair_mace.lmp         # MACE ML-IAP pair_style and element mapping (Pt C O H F S)
│   │   └── pair_lj_test.lmp      # Soft Lennard-Jones dummy potential for --dry-run plumbing tests
│   └── ZPE-inital/               # Zero-point energy / initial validation scripts
│
├── analysis/                     # Post-processing, equivalence checks, and reporting
│   ├── check_equiv.py            # Validates force/energy agreement between ASE MACE and LAMMPS ML-IAP
│   ├── analyze_run.py            # Parses logs, converts trajectories to .extxyz, plots thermo/density
│   └── summarize_pipeline.py     # Aggregates status and metrics across all sweep runs into summary.tsv
│
└── pipeline_runs/                # Timestamped execution outputs (<YYYYMMDD_HHMMSS>/)
    └── <timestamp>/
        ├── pipeline.log          # Master execution log
        ├── summary.tsv           # Final performance and stability table across all systems
        └── <system_id>/          # Per-structure results (md.log, md.extxyz, plots, status.txt)
```

---

## ⚙️ Detailed Breakdown: What `run_pipeline.sh` Does

`run_pipeline.sh` is an end-to-end bash orchestrator designed to build structures, verify model accuracy, execute GPU-accelerated MD simulations in parallel, and summarize degradation metrics.

### 1. Configuration & CLI Flags
At startup, the script loads defaults from environment variables (which you can override inline when calling the script) and parses three flags:
* `--dry-run`: Replaces the MACE potential with a fast, soft Lennard-Jones potential (`pair_lj_test.lmp`), lowers the temperature/timestep, and skips ASE equivalence checks. Used strictly to verify file paths, dumps, and script plumbing.
* `--skip-build`: Skips structure generation and reuses the existing `structure_gen/structures_sweep/manifest.txt`.
* `--pilot`: Appends an optional post-equilibration stage using **PLUMED** (`prep_plumed.py` + `fix plumed`) to test collective variables on specific Pt sites (`vertex`, `edge`, `facet`, `perimeter`). *Note: Requires a LAMMPS binary compiled with the `PLUMED` package.*

### 2. Preflight Checks
Before running any physics, the script verifies:
* Required binaries exist (`python`, `lmp`, `packmol`, and `plumed` if `--pilot` is active).
* All required Python scripts, LAMMPS input files, and MACE `.model`/`.pt` files are present.
* Core Python dependencies (`numpy`, `scipy`, `ase`, `matplotlib`) can be imported cleanly.

### 3. Structure Sweep Generation (`sweep_build.py`)
Unless `--skip-build` is set, the script invokes `sweep_build.py` to generate Pt nanoparticles (octahedra of varying nm diameters) on graphene, optionally decorated with surface oxygen (`COVERAGES`), solvated with water and Nafion chains via **Packmol**, and writes a list of all generated `data.lammps` paths to `manifest.txt`.

### 4. Parallel GPU Execution (`run_one`)
The script reads `manifest.txt` and distributes systems across the GPU IDs specified in `GPUS` (e.g., `GPUS="0 1"` runs two systems concurrently, one per GPU). For each system, it executes four sub-steps inside `pipeline_runs/<timestamp>/<system_id>/`:

* **Step 2a — ASE vs. LAMMPS Equivalence (`check_equiv.py`):**
  Depending on `EQUIV` (`first`, `all`, or `none`), evaluates the initial structure and rattled copies using both Python ASE (`MACECalculator`) and LAMMPS (`run 0`). Ensures force discrepancies are within tolerance (`< 1e-3 eV/Å`) before trusting MD results.
* **Step 2b — Plain MD Simulation (`ptgr_naf.lmp`):**
  Runs LAMMPS with Kokkos GPU acceleration (`-k on g 1 -sf kk`) through three stages:
  1. **Minimization:** FIRE minimizer (`nmin` steps) to remove bad contacts.
  2. **Heating:** Velocity initialization at `100 K` and NVT ramp to `353 K` (`nheat` steps).
  3. **Equilibration:** NVT dynamics at `353 K` (`neq` steps), saving `md_equil.data` and `md_equil.restart`.
  * *Trajectory Conversion:* If `EXTXYZ=0` (when LAMMPS lacks the `EXTRA-DUMP` package), `md.lammpstrj` and `md_energy.dat` are automatically converted to `md.extxyz` via ASE.
* **Step 2c — Automated Analysis (`analyze_run.py`):**
  Parses `md.log` for simulation speed (`steps/s` and `ns/day`) and analyzes `md.extxyz` to track Pt coordination numbers, detached Pt atoms, Nafion bond-breaking events, and z-density profiles (`md_thermo.png`, `md_zdensity.png`, `md_traj_metrics.csv`).
* **Step 2d — Optional PLUMED Pilot:**
  If `--pilot` is passed, restarts from `md_equil.restart` and runs `PILOT_STEPS` with PLUMED active to verify CV tracking (`COLVAR`).

### 5. Pipeline Summary (`summarize_pipeline.py`)
Once all background GPU jobs finish, `summarize_pipeline.py` compiles every system's `status.txt` and metrics into a single master table (`summary.tsv`) printed to the terminal and saved in the run folder.

---

## 🚀 Usage Examples

### Standard MACE MD Sweep (No PLUMED)
Run plain minimization, heating, and NVT equilibration using the MACE V9 model:
```bash
conda activate lmp_mace
EXTXYZ=0 bash run_pipeline.sh
```

### Customizing System Sizes, Coverages, and Step Counts
Override any configuration variable inline without editing the script:
```bash
EXTXYZ=0 SIZES="0.8,1.2" COVERAGES="0.0,0.375" NMIN=1000 NHEAT=2000 NEQ=5000 bash run_pipeline.sh
```

### Multi-GPU Execution
Distribute systems across multiple GPUs simultaneously (1 system per GPU):
```bash
EXTXYZ=0 GPUS="0 1" bash run_pipeline.sh
```

### Reusing Existing Structures (`--skip-build`)
Skip Packmol structure generation and immediately run MD on the structures already listed in `structure_gen/structures_sweep/manifest.txt`:
```bash
EXTXYZ=0 bash run_pipeline.sh --skip-build
```

### Plumbing Dry Run (`--dry-run`)
Test file generation, trajectory conversion, and plotting scripts in seconds using a dummy Lennard-Jones potential:
```bash
EXTXYZ=0 bash run_pipeline.sh --dry-run
```

---

## 🛠️ Environment Notes & Troubleshooting

* **LAMMPS Packages:** Your `lmp_mace` Conda environment includes `KOKKOS`, `ML-IAP`, `ML-SNAP`, and `PYTHON`. Because `EXTRA-DUMP` and `PLUMED` are not compiled into this binary:
  * Always keep `EXTXYZ=0` so LAMMPS writes `.lammpstrj` and lets Python convert it to `.extxyz`.
  * Omit `--pilot` unless you install a PLUMED-enabled LAMMPS build.
* **Key Summary Flags (`summary.tsv`):**
  * `Pt_detached_max > 0`: Indicates a Pt atom leached/detached from the nanoparticle during unbiased MD.
  * `Nafion_broken_max > 0`: Indicates C–F, C–C, or C–S bond dissociation in the Nafion ionomer.
  * `steps_per_s` / `ns_per_day`: Benchmarks GPU throughput to estimate wall-clock time for long production runs.
```

### Quick Recap of What `run_pipeline.sh` Does:
1. **Preflight:** Checks that your Conda environment (`lmp_mace`), `packmol`, input templates, and MACE `.model`/`.pt` files exist.
2. **Build (`sweep_build.py`):** Generates Pt nanoparticles of specified diameters on graphene, adds O adsorbates if requested, packs water and Nafion around them, and records the paths in `manifest.txt`.
3. **Equivalence (`check_equiv.py`):** Compares energies and forces between Python ASE and LAMMPS ML-IAP on the first structure (`EQUIV=first`) to guarantee the model loaded properly.
4. **MD Simulation (`ptgr_naf.lmp`):** Runs Kokkos GPU-accelerated MD (FIRE minimization $\rightarrow$ heating ramp to $353\text{ K}$ $\rightarrow$ NVT equilibration), then converts the custom trajectory into `md.extxyz`.
5. **Analysis & Summary:** Plots thermodynamic stability and z-density profiles, checks for Pt detachment or Nafion bond breaking, and prints `summary.tsv`.

Let me know if you'd like to adjust any default step counts or add SLURM batch submission templates to the documentation!


Which outputs matter

Fifteen picoseconds is far too short to equilibrate anything, so these outputs only catch failures. Open them in this order:

summary.tsv.
equiv: should say PASS.
steps_per_s: sets the budget.
Nafion_broken_max: must be 0.
Pt_detached_max: must be 0 without bias.
min_S_Pt_A: the pt head mode should be clearly smaller than the graphene mode. If not, the head placement isn’t doing what you intended.
md_thermo.png. Temperature should ramp and then plateau, and potential energy should have no spikes.
md_traj_metrics.png. Mean Pt–Pt coordination and Pt–C contacts should be steady, and a falling Pt coordination suggests melting or reconstruction. Check the sulfonate-on-Pt and water-on-Pt panels against the head mode.
md_zdensity.png. Look for water layering near the graphene and Pt, and a flat middle.
pilot/. The COLVAR plot and its SIGMA advice.

What a full run produces
Data	Where it comes from	What you can claim from it
Equivalence, speed, stability	equiv.log, summary.tsv	The LAMMPS model matches the validated ASE model. Supports the budget and the SI.
Structure metrics, density profiles, RDFs	md_traj_metrics.*, md_zdensity.png, mace_md.py rdf	The model keeps Pt, Nafion and water sensible. Compare with short AIMD.
Unbiased CV fluctuations	pilot COLVAR	Sets SIGMA and shows that the CVs behave.
Metadynamics CV time series	COLVAR (target Pt–Pt and Pt–O coordination, O-type splits, height, S–Pt, Pt–C)	Event ordering, recrossings, whether detachment is reversible.
Free-energy surfaces	HILLS, analyze_run.py fes	Detachment barrier and attached-versus-detached ΔF per condition.
Convergence	ΔF versus simulated time	Whether a barrier is quotable at all.
Saddle-region frames	merge + extract_frames.py --where	The mechanism: which O type coordinates the target Pt at the saddle.
DFT single points on those frames	your CP2K runs	Model error exactly where the claim lives.
What it can’t give you
Absolute rates. At 353 K, 0.1 eV is about a factor of 50 in rate.
Potential dependence. Coverage maps to potential only through your CHE argument.
Statistics from one trajectory. You need independent walkers and Nafion packings for error bars.
A full dissolution product. I’d define the endpoint as detachment or lifting of the target Pt, not a solvated Pt ion.