# analyze_run.py — what it does and how to read it

`analysis/analyze_run.py` is a toolbox of seven sub-commands for looking at what a LAMMPS (and optionally PLUMED)
run of the **Pt nanoparticle / graphene / Nafion / water** system did. You give it a file from the run
(the LAMMPS log, the trajectory, the PLUMED `COLVAR` or `HILLS` file); it prints numbers, writes plots and CSV files,
and prints plain-language warnings.

It **does not run any simulation**. It only reads files, so you can run it again at any time, on any finished run,
without touching the MD.

```
python analyze_run.py <sub-command> <input file(s)> [options]
```

| Sub-command | Input | What you learn |
|---|---|---|
| [`log`](#1-log) | `md.log` | Speed, temperature, energy drift |
| [`traj`](#2-traj) | `md.extxyz` | Pt/graphene/Nafion/water behaviour frame by frame |
| [`bonds`](#3-bonds) | `md.extxyz` | Are Nafion bonds stretching normally or breaking? |
| [`colvar`](#4-colvar) | PLUMED `COLVAR` | How the collective variables moved; advice for SIGMA |
| [`fes`](#5-fes) | PLUMED `HILLS` | Free-energy surface, barrier, convergence |
| [`merge`](#6-merge) | `.extxyz` + `COLVAR` | Attach CV values to each frame (to pick frames by CV) |
| [`convert`](#7-convert) | `.lammpstrj` | Make an extxyz if LAMMPS could not write one |

Requirements: Python with `numpy`, `scipy`, `ase`, `matplotlib` (the `lmp_mace` env has them).
Units everywhere: **Å, eV, ps, K**.

---

## Words used in this document

- **Frame**: one snapshot of all atom positions stored in the trajectory.
- **Pt–Pt coordination number (CN)**: how many other Pt atoms are within 3.2 Å of a Pt atom. Bulk Pt has 12, a surface atom about 9 on a flat facet, a corner atom 4–6. A Pt atom that leaves the particle has CN → 0.
- **CV (collective variable)**: a number computed from atom positions that PLUMED uses to describe progress (here: the target Pt atom's Pt–Pt coordination, `cn_ptpt`, and its Pt–O coordination, `cn_pto`).
- **COLVAR**: the PLUMED text file with the CVs printed every few steps. **HILLS**: the file listing each Gaussian "hill" that metadynamics added.
- **Metadynamics / bias**: PLUMED adds small repulsive hills in CV space so that the system leaves states it has already visited and crosses barriers. From the hills one reconstructs the free energy.
- **Step vs time**: time (ps) = MD step × timestep (0.0005 ps = 0.5 fs). `ptgr_naf.lmp` restarts the step counter after the minimisation, so step, time and PLUMED time all agree.

---

## Files the script expects

| File | Produced by | Notes |
|---|---|---|
| `md.log` | LAMMPS `-log md.log` | Must contain the thermo table (`Step Time Temp PotEng ...`) and `Performance:` lines |
| `md.extxyz` | `dump extxyz` in `ptgr_naf.lmp` | Needs the info key `Timestep` (LAMMPS writes it). Atom order must be the data-file order |
| `COLVAR`, `HILLS`, `HILLS.0` ... | PLUMED | `HILLS.N` = one file per walker |

**Atom-group assumptions (all inferred from the FIRST frame).** `traj` and `bonds` rebuild the groups themselves:

- **Graphene C** = all C atoms that come *before the first Pt* in the file (the builder writes graphene C first). Override with `--n-graphene N` if your ordering differs.
- **Nafion atoms** = every molecule (connected set of covalently bonded atoms) that contains an F or S atom.
- **Sulfonate O** = O within 1.9 Å of an S atom.
- **Adsorbed O** = O that is not sulfonate, has a Pt within 2.5 Å, and has at most one H.
- **Water O** = every other O not in Nafion.

These roles are fixed by atom index for the whole trajectory. If a water molecule later binds to Pt it is still counted as "water O" (it will show up in the `n_Ow_on_Pt` metric).

---

## 1. `log`

**In one sentence:** reads the LAMMPS log and tells you how fast the run was and whether the temperature and energy behaved.

```
python analyze_run.py log md.log [--nprod 4000000] [--dt-fs 0.5] [--out md]
```

**What it does**

1. Finds every thermo table (one per `minimize`/`run` command) and every `Performance:` line.
2. Prints speed for each `run`: steps/s, ns/day, hours per ns.
3. If you pass `--nprod N`, prints how long N production steps would take at the last measured speed.
4. Plots temperature and potential energy for all blocks end to end.
5. For the **last block** prints the mean ± sd of temperature over its second half and the **potential-energy drift** (last row minus first row of the block), also per atom and per ps when the atom count can be read from the log.

**Output:** `<out>_thermo.png`, plus the printed lines.

**How to read it**

- `last run speed` is the number that sets your production budget. 1 ns = 2×10⁶ steps at 0.5 fs.
- Temperature should sit at the target (353 K) with a spread of a few percent for ~1,000 atoms.
- A large negative potential-energy drift (for example −10 eV over 100 ps for ~800 atoms, which is about 12 meV/atom and 0.1 eV/ps) means the system is **still relaxing**: it is not equilibrated yet. A converged block drifts by about as much as its own fluctuations.
- NVT does not conserve total energy (the thermostat adds and removes energy), so do not look for that.

**Caveat:** only the thermo tables are read. The timing and histogram tables LAMMPS prints at the end of each run are ignored (an early version of `summarize_pipeline.py` mis-read them and reported a wrong temperature; that is fixed).

---

## 2. `traj`

**In one sentence:** goes through every frame of the trajectory and computes a set of numbers describing the Pt particle, its contacts with graphene, oxygen and sulfonate, and the health of the Nafion, then plots them against time.

```
python analyze_run.py traj md.extxyz [--out md] [--dt 0.0005] [--start 0] [--stride 1]
                           [--n-graphene N] [--bulk-lo 8] [--bulk-hi 20]
```

**What it does**

1. Reads the first frame and builds the atom groups (see above).
2. For each frame computes the metrics in the table below.
3. Writes them to a CSV and plots them.
4. Averages the water O height distribution over the **second half** of the frames and plots a water density profile.
5. Prints, for every metric, the mean over the **first half**, the **third quarter** and the **fourth quarter** of the frames, and the sd of the fourth quarter. A `*` marks a metric whose last two quarters differ by more than that sd: it is **still evolving at the end of the run** (a change that happened early and then stopped, such as a contact count settling, is not flagged). It also prints the water per sulfonate group (λ) and a `FLAGS:` line.

**Outputs**

| File | Contents |
|---|---|
| `<out>_traj_metrics.csv` | One row per frame, all metrics below |
| `<out>_traj_metrics.png` | 3×3 panels vs time |
| `<out>_zdensity.png` | Water density vs height above the top graphene layer |

**Metric dictionary** (CSV column → meaning)

| Column | Meaning | Typical / what to look for |
|---|---|---|
| `pt_cn_mean` | Mean Pt–Pt coordination over all Pt atoms (cutoff 3.2 Å) | Ideal 19-atom octahedron ≈ 6.3. A steady drop = the particle is loosening/reconstructing |
| `pt_cn_min` | Lowest Pt–Pt CN of any Pt atom | Drops to 0 if an atom leaves |
| `pt_detached` | Number of Pt atoms with CN = 0 | Must be 0 in plain MD at 353 K |
| `pt_rg` | Radius of gyration of the Pt atoms (Å, no periodic wrapping) | Grows if the particle spreads or loses atoms |
| `pt_c_contacts` | Number of Pt atoms with a graphene C within 2.8 Å | Anchoring of the particle to the support |
| `pt_c_min` | Shortest Pt–C distance | ~1.9–2.2 Å for bonded Pt |
| `pt_s_min` | Shortest Pt–S distance (any Pt, any S) | **9.9 means "no S within 3.6 Å"** (sentinel value). ~3 Å = sulfonate bound through O |
| `n_Os_on_Pt` | Number of sulfonate O within 2.7 Å of any Pt | Direct sulfonate contact |
| `n_Oads_on_Pt` | Number of adsorbed O/OH within 2.7 Å of any Pt | Coverage check |
| `n_Ow_on_Pt` | Number of water O within 2.7 Å of any Pt | Water contact |
| `pt_O_cn_mean` | Mean number of O atoms (any type) within 2.7 Å per Pt | Overall oxygen environment |
| `nafion_broken` | Tracked Nafion bonds that are stretched past a threshold (total) | See next two rows |
| `nafion_broken_OH` | …of which O–H bonds (sulfonic-acid proton leaving to water) | Expected: protons transfer to water. Compare with the number of sulfonate groups |
| `nafion_broken_other` | …of which anything else (C–F, C–S, S–O, C–O, C–C) | **Should be 0.** Non-zero means backbone/side-chain damage |
| `step`, `time_ps`, `E_eV` | Step, time, potential energy of the frame | `E_eV` is blank (nan) if the dump has no energy |

The bond threshold is *length > 1.3 × (length at frame 0) + 0.25 Å*. Bonds are those found at frame 0.

**The `FLAGS:` line**

- `Pt atom(s) with no Pt neighbour from t = …` — a Pt atom left the particle.
- `Nafion bonds other than O-H broken …` — real damage to the polymer.
- `info: up to N sulfonic-acid O-H bonds dissociated` — proton transfer; informational, not an error.

**The water density line**

The profile counts water O in 0.25 Å height bins, divided by the cell *area* (not volume), converted to g/cm³. The dotted line is 0.97. Because the cell includes vacuum and the Pt particle's volume, the number is a *sanity check*, not a bulk density. For a thick water layer you want a flat plateau between layering peaks near the graphene. **If there are few waters** (for example 75 molecules ≈ a film of about 5 Å at liquid density) there is no bulk-like region at all; the script then prints a NOTE and the density line should be ignored. In that case the water per sulfonate group (λ) is the meaningful number.

**Caveats**

- All cutoffs (3.2, 2.7, 2.8, 3.6, 2.5, 1.9 Å) are hard-coded in the script.
- A short run (a few ps) mostly reflects the starting packing, not equilibrium: compare the first and second half before reading anything into contact numbers.

---

## 3. `bonds`

**In one sentence:** measures the lengths of every Nafion bond over the trajectory, grouped by bond type, so you can tell thermal stretching from bond breaking.

```
python analyze_run.py bonds md.extxyz [--out md] [--start 0] [--stride 1] [--n-graphene N]
```

**What it prints (one row per bond type, e.g. `C-F`, `C-S`, `O-S`, `C-O`, `C-C`, `H-O`)**

| Column | Meaning |
|---|---|
| `N` | Number of bonds of this type |
| `mean A`, `sd` | Mean and spread of all bond lengths over all frames |
| `99.9%`, `max` | The 99.9th percentile and the largest length seen |
| `frames with a bond > mean+0.3 A` | Fraction of frames in which *any* bond of this type is more than 0.3 Å longer than average |
| `worst (step, atoms)` | Step and atom numbers (1-based, as in the data file) of the longest bond, so you can open exactly that frame in a viewer |

**Output:** `<out>_bonds.png` (one log-scale histogram per bond type).

**How to read it**

- A bond that stretches and comes back is thermal motion if its maximum stays within a few tenths of an Å of the mean.
- A long tail (S–O or C–S lengths above about 2 Å) means the model lets the bond break and re-form: a model problem to investigate.
- Viewers draw a bond only below a distance cutoff, so a slightly stretched bond can look "detached". This command gives the actual numbers.
- `H-O` bonds belong to sulfonic acid groups; large values there are proton transfer, not damage.

---

## 4. `colvar`

**In one sentence:** summarises a PLUMED `COLVAR` file: how each CV fluctuated, whether the target Pt left, and (for unbiased runs) what hill width SIGMA to use.

```
python analyze_run.py colvar COLVAR [--out run] [--skip-ps 0] [--detach-cn 0.5]
```

**What it does**

1. Reads the `#! FIELDS` header and all rows.
2. Skips the first `--skip-ps` picoseconds **of this file** (relative to its first row, not absolute time; after a restart PLUMED time continues from the MD step counter). If skipping would leave fewer than 5 rows it uses all rows and says so.
3. Warns if the file covers less than 10 ps.
4. Prints mean, sd, min, max of every column.
5. Reports when `cn_ptpt` first drops below `--detach-cn` and the fraction of time spent below it.
6. **Unbiased files only** (no `metad.bias` column): prints suggested SIGMA = half the sd of `cn_ptpt` and `cn_pto`, floored at 0.1, and the matching `prep_plumed.py --sigma` argument.
7. Plots every available CV against time, and a 2-D histogram of `cn_ptpt` vs `cn_pto`.

**Outputs:** `<out>_colvar.png`, `<out>_colvar2d.png`.

**CV names you will see** (written by `prep_plumed.py`)

| CV | Meaning |
|---|---|
| `cn_ptpt` | Target Pt's coordination to the other Pt atoms (biased in metadynamics) |
| `cn_pto` | Target Pt's coordination to all O (biased in metadynamics) |
| `cn_ptoa`, `cn_ptow`, `cn_ptos` | …split by O type: adsorbed, water, sulfonate (monitored only) |
| `cn_pts` | Coordination to S |
| `dz.z` | Height of the target above the top graphene layer (Å) |
| `dS.min`, `dC.min` | Closest S / closest graphene C to the target (Å) |
| `metad.bias`, `metad.rbias` | Bias energy and its reweighting version (metadynamics runs only) |

**Caveats**

- The sd values of a **biased** run are not natural fluctuations; do not use them for SIGMA. The 2-D histogram of a biased run is **not** a free energy (use `fes`).
- SIGMA advice from a run shorter than ~20–50 ps is unreliable (the script warns).

---

## 5. `fes`

**In one sentence:** rebuilds the free-energy surface from the hills metadynamics deposited, and shows how it changed over the run so you can judge convergence.

```
python analyze_run.py fes HILLS [HILLS.1 ...] --bias-factor 15 [--temp 353] [--out run]
                              [--xmin -1 --xmax 12 --ymin -1 --ymax 10 --step 0.05]
                              [--slices 8] [--attached-cn 3.0] [--detached-cn 1.0] [--fmax 3.0]
```

**What it does**

1. Reads one or more `HILLS` files (give all walkers' files; hills are merged and sorted by time). Requires exactly **two** CVs: the first is treated as the "detachment" coordinate (`cn_ptpt`).
2. Sums all Gaussians on a grid. For well-tempered metadynamics the free energy is `F = −γ/(γ−1) × V_bias` (γ = `--bias-factor`, must match the run), shifted so the minimum is 0.
3. Does this cumulatively for `--slices` equal slices of the simulated time.
4. For each slice reports:
   - **ΔF detached − attached**: free energy of the region `first CV < --detached-cn` relative to `first CV > --attached-cn`, integrated over the second CV.
   - **barrier**: height of the 1-D projection (along the first CV) between the attached minimum and the detached minimum, measured from the attached minimum.
5. Prints a convergence statement: how much ΔF and the barrier changed over the last 3 slices.

**Outputs**

| File | Contents |
|---|---|
| `<out>_fes.png` | Left: 2-D free-energy map at the end of the run (colour clipped at `--fmax`). Middle: 1-D projection for every slice (early→late colours). Right: ΔF detached−attached vs simulated time |
| `<out>_fes.npz` | Final grid: `gx`, `gy`, `F`, `cv` names, for your own plots |

**How to read it**

- The **right-hand panel is the key one**: if ΔF still drifts by more than about 0.1 eV over the last slices, the run is **not converged and no barrier should be quoted**.
- A healthy run crosses between attached and detached **many times** (check `cn_ptpt` in the `colvar` plot). A trajectory that leaves and never returns gives an unconverged surface.
- The 1-D barrier is a crude estimate; it depends on the choice of basins (`--attached-cn`, `--detached-cn`).

**Caveats**

- PLUMED's default hill shape is slightly different from the plain Gaussians used here; the difference is negligible for this purpose.
- The surface is only as good as the model and the CVs. The mechanism is found by looking at structures in the saddle region (see `merge`).

---

## 6. `merge`

**In one sentence:** copies the CV values from `COLVAR` onto the matching frames of an extxyz, so you can select frames by CV value later.

```
python analyze_run.py merge metad.extxyz COLVAR --out metad_cv.extxyz [--dt 0.0005] [--colvar-stride 100]
```

For every frame it computes time = `Timestep × dt`, finds the nearest `COLVAR` row, and stores each column as an info key (`.` replaced by `_`: `cn_ptpt`, `dz_z`, `dS_min`, …). It also stores `md_step`, `time_ps` and, if the trajectory has `Potential_energy`, an `energy` label.

A frame without a `COLVAR` row within 0.6 × `dt` × `--colvar-stride` gets no CV keys (the equilibration frames before PLUMED started). The command prints how many.

**Why:** the next step is `extract_frames.py ... --where cn_ptpt:2.0:3.5` to pull frames near a barrier top for DFT single points or mechanism analysis.

---

## 7. `convert`

**In one sentence:** rewrites a LAMMPS `.lammpstrj` as an extxyz (with forces and the `Timestep` key) if your LAMMPS has no `dump extxyz`.

```
python analyze_run.py convert md.lammpstrj --out md.extxyz [--specorder Pt,C,O,H,F,S]
```

`--specorder` is the element for LAMMPS atom type 1, 2, 3, …; it must match the data file. `run_pipeline.sh` calls this automatically when `md.extxyz` is missing.

---

## Typical workflows

**A. After a plain MD run (equilibration check)**

```
python analyze_run.py log md.log
python analyze_run.py traj md.extxyz --out md --dt 0.0005
python analyze_run.py bonds md.extxyz --out md
```

Read, in this order: the `FLAGS:` line, `md_thermo.png` (temperature plateau, no energy spikes), the energy drift, `md_traj_metrics.png`, the bond table.

**B. Unbiased PLUMED pilot → choose SIGMA**

```
python analyze_run.py colvar pilot/COLVAR --skip-ps 2 --out pilot/pilot
```

Use the printed `--sigma a,b` in `prep_plumed.py --mode metad`. The pilot should span at least 20–50 ps.

**C. After a metadynamics run**

```
python analyze_run.py log metad.log --out metad
python analyze_run.py colvar COLVAR --out metad
python analyze_run.py fes HILLS --bias-factor 15 --temp 353 --out metad
python analyze_run.py merge metad.extxyz COLVAR --out metad_cv.extxyz
```

**D. Re-analyse every system folder of a pipeline run** (no MD is re-run)

```
for d in cov*/; do ( cd "$d" && python $ROOT/analysis/analyze_run.py traj md.extxyz --out md --dt 0.0005 > analyze_traj.txt 2>&1 ); done
python $ROOT/analysis/summarize_pipeline.py .
```

---

## Troubleshooting

| Message / symptom | Cause | Fix |
|---|---|---|
| `no thermo blocks found` | Log is not a LAMMPS log, or thermo style differs | Run LAMMPS with `-log file`; header must contain `Step … Temp … PotEng` |
| Time axis of `traj` looks wrong (starts at 0, steps by one frame) | Trajectory has no `Timestep` key, so the frame index is used instead | Use `dump extxyz` from `ptgr_naf.lmp`, or `convert` from lammpstrj (which sets `Timestep`) |
| `--skip-ps` leaves nothing | File shorter than the skip | The script now falls back to all rows and warns |
| `pt_s_min = 9.9` | No S within 3.6 Å (sentinel) | Not an error |
| Wrong graphene/Nafion counts printed on the first line | Atom order differs from graphene → Pt → Nafion → water | Use `--n-graphene N` |
| `expected 2 CVs in HILLS` | `HILLS` from a run with a different CV set | `fes` is written for the two-CV setup of `prep_plumed.py` |
| `nafion_broken_other > 0` | Genuine bond breaking (or very distorted geometry) | Run `bonds`, open the reported frame, check against DFT |

## Limits to keep in mind

- This script **describes** the simulation. It cannot tell you whether the model is *accurate*; for that compare with DFT (`extract_frames.py` → CP2K) and use `model_disagreement.py`.
- Cutoffs, thresholds and group definitions are fixed in the source; edit them there if your system differs.
- Plain-MD metrics from a few ps mostly reflect the starting structure.