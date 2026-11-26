# run_pipeline.sh — what it does and how to use it

`run_pipeline.sh` (repository root) is a **bash script that chains the whole test procedure**:
build structures → check the model → run a short MD of every structure → analyse → (optionally) run a short unbiased PLUMED
pilot → print one summary table. It is a *smoke test and screening tool*: it tells you whether each system builds, runs
and behaves sensibly with the MACE model in LAMMPS, and how fast it is. It does **not** run production metadynamics.

```
./run_pipeline.sh [--dry-run] [--skip-build] [--pilot] [-h]      # everything else is an environment variable
```

---

## The big picture

```
 preflight  ──►  1. build structures  ──►  pretest (1 step)  ──►  per system (one per GPU at a time)  ──►  summary
 (checks)        sweep_build.py            fail fast                2a equivalence  2b plain MD            summarize_pipeline.py
                                                                    2c analysis     2d optional pilot
```

### Preflight — before anything expensive

Stops with a clear message if something is missing:

- commands: `python`, the `lmp` binary, `packmol` (unless `--skip-build`), `plumed` (for a real pilot);
- files: `sweep_build.py`, `build_interface.py`, `prep_plumed.py`, `analyze_run.py`, `check_equiv.py`, `summarize_pipeline.py`, the LAMMPS input and pair file;
- the two model files (real runs only);
- **model format**: `MODEL_LMP` must be the ML-IAP format. A TorchScript model (made without `--format=mliap`) is detected and rejected with the conversion command;
- Python modules `numpy`, `scipy`, `ase`, `matplotlib`;
- **which LAMMPS it is**: it runs `lmp -h` and logs the path and the compiled-in packages
  (`KOKKOS= ML-IAP= PYTHON= PLUMED= COLVARS=`). If `lmp -h` does not even start (a `GLIBC` error usually means a pip `lammps` wheel is shadowing your real build) it stops. If PLUMED is missing the pilot is **disabled with a warning** instead of failing later.

### 1. Build structures

Calls `structure_gen/sweep_build.py`, which drives `build_interface.py` over every combination of
`SIZES × COVERAGES × SITES × SEEDS × HEADS`. Each *condition* gets its own folder, e.g.
`structures_sweep/cov0.250_mixed_s426_pt/octahedron_055pt_O0.261ML_s426/data.lammps`.

- Coverage 0.0 builds one bare system per head mode (site and seed are ignored).
- `sweep_build.py` writes `structures_sweep/manifest.txt`: the list of `data.lammps` files **built by this call**.
  (Older structures in the same folder are not listed, so they are not re-tested.)
- With `--skip-build` (or `SKIP_BUILD=1`) this stage is skipped and the **existing manifest from the last build** is used.
- `ONLY='regex'` keeps only manifest entries whose full path matches the regex.

### Pretest — fail fast

Runs LAMMPS once on the **first** structure for a single MD step. This loads the model, the pair style and Kokkos.
If it fails the sweep stops at once with the error and a hint (for example "model is not in ML-IAP format")
instead of failing 20 times. Skip with `SKIP_PRETEST=1`.

### Per-system test (`run_one`) — runs once per structure

Each structure runs in its own folder `OUT/<condition>__<structure>/`, on the GPU assigned to it.

| Step | What happens | Files |
|---|---|---|
| **2a. Equivalence** | `check_equiv.py` compares energies/forces from ASE-MACE and LAMMPS on the structure and two rattled copies. `EQUIV=first` (default) does this only for the first system; `all`; `none`. PASS if max force difference < 1e-3 eV/Å | `equiv.log`, `equiv_files/` |
| **2b. Plain MD** | `ptgr_naf.lmp`: minimise (`NMIN`), heat 100 K → 353 K (`NHEAT` steps), NVT equilibrate (`NEQ` steps), no PLUMED, no production | `md.log`, `md.screen`, `md.stdout`, `md.lammpstrj`, `md.extxyz`, `md_energy.dat`, `md_min.data`, `md_equil.data`, `md_equil.restart` |
| **2c. Analysis** | `analyze_run.py log` and `traj` (see `README_analyze_run.md`) | `analyze_log.txt`, `analyze_traj.txt`, `md_thermo.png`, `md_traj_metrics.csv/.png`, `md_zdensity.png` |
| **2d. Pilot** (only with `PILOT=1`) | `prep_plumed.py` picks a target Pt of class `vertex` (then `edge`, `facet`, `perimeter` until one works), writes an **unbiased** PLUMED file, and LAMMPS runs `PILOT_STEPS` steps from `md_equil.restart` with PLUMED monitoring the CVs; `analyze_run.py colvar` summarises | `prep.log`, `pilot/` (`plumed.dat`, `COLVAR`, `pilot.log`, `analyze_colvar.txt`, `pilot_colvar*.png`, …) |
| **status** | One line summarising the outcome | `status.txt` |

If `md.extxyz` was not written (LAMMPS without `dump extxyz`) it is created from `md.lammpstrj` with `analyze_run.py convert`.
The log analysis always projects the cost of 4,000,000 production steps (`--nprod 4000000`, fixed in the script).

### Running in parallel

`GPUS="0 1"` means two systems at a time, one per GPU (`CUDA_VISIBLE_DEVICES` is set per job; each LAMMPS uses its single visible GPU).
Systems are started in **chunks of as many as there are GPUs** and the script waits for the whole chunk to finish before starting the next
(so a slow system leaves the other GPU idle for a while). Check `nvidia-smi` that the GPUs you list are free.

### Summary

`summarize_pipeline.py OUT` reads every system folder and prints/writes `OUT/summary.tsv`. It exits non-zero if any system's MD
failed, an equivalence test failed or a requested pilot failed.

| Column | Meaning |
|---|---|
| `system`, `atoms` | Condition and structure name; number of atoms |
| `build` | `ok` if the structure existed |
| `equiv` | `PASS (max force difference)` / `FAIL` / blank if not run |
| `md` | `ok` or `FAILED` |
| `steps_per_s`, `ns_per_day` | Speed of the **last** LAMMPS run (the equilibration). Sets the production budget |
| `T_end_K` | Mean temperature over the second half of the last thermo block (target 353 K) |
| `Pt_detached_max` | Largest number of Pt atoms with no Pt neighbour in any frame; should be 0 |
| `backbone_broken_max` | Nafion bonds **other than O–H** that broke; should be 0 |
| `SO3H_dissoc_max` | Sulfonic-acid O–H bonds dissociated (proton transfer to water): expected, compare with the number of sulfonate groups |
| `S_Pt_contact_frac` | Fraction of frames (second half) with any S within 3.6 Å of any Pt |
| `Os_on_Pt_mean` | Mean number of sulfonate O within 2.7 Å of Pt (second half) |
| `Pt_CN_mean` | Mean Pt–Pt coordination (second half) |
| `pilot` | `ok(site)` / `FAILED` / `-` |
| `note` | Collected warnings (`equiv_failed;`, `traj_analysis_failed;`, `pilot_failed;`) |

A `DRY_RUN` marker in the output folder makes the summary print a banner: dry-run numbers describe the dummy potential, not the model.

---

## Output folder layout

```
pipeline_runs/<timestamp>/
├── pipeline.log                 everything the script printed
├── summary.tsv                  the table above
├── _pretest/                    the one-step model-load test
└── <condition>__<structure>/    one per system
    ├── status.txt  equiv.log  equiv_files/
    ├── md.log  md.screen  md.stdout  md.lammpstrj  md.extxyz  md_energy.dat
    ├── md_min.data  md_equil.data  md_equil.restart
    ├── analyze_log.txt  analyze_traj.txt  md_thermo.png  md_traj_metrics.csv/.png  md_zdensity.png
    └── pilot/                   only with PILOT=1
```

---

## Settings

Every setting is an environment variable with a default in the CONFIG block of the script. Values given on the
command line win: `SIZES=0.8,1.2 COVERAGES=0.0,0.5 ./run_pipeline.sh`.
(The defaults below are those of the shipped script; **your copy may have different ones**. Put your own in `pipeline.env`, see below.)

| Variable | Default | Meaning |
|---|---|---|
| `ROOT` | folder of the script | Repo root |
| `PYTHON` | `python` | Python used for all scripts |
| `STRUCT_DIR`, `ANALYSIS_DIR`, `LMP_DIR` | `$ROOT/structure_gen`, `/analysis`, `/lammps/ptgr_naf` | Script folders |
| `LMP_INPUT` | `$LMP_DIR/ptgr_naf.lmp` | The LAMMPS input |
| `MODEL_ASE` | `mace_V9_…model` | Model for ASE (equivalence check) |
| `MODEL_LMP` | `…-mliap_lammps.pt` | **ML-IAP-format** model for LAMMPS |
| `HEAD` | empty | Head name for multihead models |
| `SIZES` | `0.8,1.0,1.15,1.3,1.5` | Volume-equivalent particle diameters, nm |
| `COVERAGES` | `0.0,0.375` | O coverage, ML per exposed surface Pt |
| `SITES` | `mixed` | O site type: `fcc\|hcp\|hollow\|4fold\|bridge\|ontop\|mixed` |
| `SEEDS` | `7` | Seeds for adsorbate placement and packing |
| `HEADS` | empty | Nafion head mode(s): `graphene`, `pt`, `free` (empty = builder default) |
| `SETS` | empty | Extra builder `Config` overrides, space-separated, e.g. `"n_nafion=2 margin=7.0"` |
| `STRUCT_OUT` | `$STRUCT_DIR/structures_sweep` | Where structures and `manifest.txt` live |
| `LMP` | `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half` | Launch command (binary + Kokkos flags) |
| `GPUS` | `0` | GPU ids, space-separated |
| `DEVICE` | `cuda` | Device for the ASE side of the equivalence test |
| `NMIN`, `NHEAT`, `NEQ` | `2000`, `5000`, `10000` | Minimiser max iterations; heating steps; equilibration steps (0.5 fs each) |
| `NDUMP` | `200` | Steps between stored frames |
| `DT` | `0.0005` | Timestep in ps; used by the analysis only (must match LAMMPS) |
| `EQUIV` | `first` | `first`, `all`, `none` |
| `EXTXYZ` | `1` | `0` if your LAMMPS has no `dump extxyz` |
| `PILOT` | `0` | `1` = run the unbiased PLUMED pilot |
| `PILOT_STEPS` | `20000` | Pilot length in steps (10 ps at 0.5 fs); use ≥ 40,000–100,000 to get a usable SIGMA |
| `PILOT_SITES` | `vertex edge facet perimeter` | Order in which target site classes are tried |
| `ONLY` | empty | Regex on the `data.lammps` path |
| `SKIP_BUILD`, `DRY_RUN`, `SKIP_PRETEST` | `0` | Same as the flags / skip pretest |
| `OUT` | `$ROOT/pipeline_runs/<timestamp>` | Output folder |
| `PIPELINE_ENV` | `$ROOT/pipeline.env` | Optional settings file |

**Command-line flags:** `--dry-run`, `--skip-build`, `--pilot` (sets `PILOT=1`, **overriding** `PILOT=0`), `-h`.

### Keeping your own defaults: `pipeline.env`

If `$ROOT/pipeline.env` exists it is sourced at the start. Use the `VAR=${VAR:-value}` form so that values given on the command line still win:

```bash
MODEL_ASE=${MODEL_ASE:-$ROOT/mace_V11_active_learning.model}
MODEL_LMP=${MODEL_LMP:-$ROOT/mace_V11_active_learning.model-mliap_lammps.pt}
COVERAGES=${COVERAGES:-0.0,0.25,0.5,0.75}
HEADS=${HEADS:-pt,graphene}
```

Updating the script then never overwrites your settings.

### Dry run

`--dry-run` replaces the MACE model by a soft dummy pair potential, lowers the temperature to 50 K and the timestep to 0.2 fs, and skips the
equivalence test. It checks files, folder structure, dumps, walls, PLUMED and restarts on any machine (it only needs a LAMMPS binary).
Nothing in its output says anything about the model.

---

## Recipes

```bash
# full default run in the background
nohup ./run_pipeline.sh > pipeline.out 2>&1 &

# plumbing test on any machine
./run_pipeline.sh --dry-run

# just the two smallest systems, with the PLUMED pilot, both GPUs
ONLY='octahedron_019pt' PILOT=1 PILOT_STEPS=40000 GPUS="0 1" ./run_pipeline.sh

# longer plain MD (about 100 ps) on a subset, reusing existing structures
NEQ=200000 NHEAT=10000 EQUIV=none PILOT=0 SKIP_BUILD=1 GPUS="0 1" \
ONLY='cov0.000_none_s426_pt|cov0.500_mixed_s426_pt' OUT=$ROOT/pipeline_runs/long_equil ./run_pipeline.sh

# re-analyse existing runs without re-running MD: see "Workflow D" in README_analyze_run.md
```

At about 30 steps/s a 100 ps run (200,000 steps) takes roughly 2 hours per system.

---

## Troubleshooting

| What you see | Meaning | What to do |
|---|---|---|
| `preflight failed` + `MISSING …` | A command/file/model is absent | Fix the listed item |
| `ERROR: MODEL_LMP … is a TorchScript/libtorch model` | The `.pt` is not ML-IAP format | `python -m mace.cli.create_lammps_model <model> --format=mliap` |
| `'lmp' … cannot run 'lmp -h'` with a `GLIBC` message | A pip `lammps` wheel is first on PATH | Set `LMP=/full/path/to/lmp -k on g 1 …` or remove the wheel |
| `built WITHOUT the PLUMED package -> PLUMED pilot disabled` | The binary has no `fix plumed` | Rebuild LAMMPS with PLUMED (`build_lammps_mace.sh`) |
| `PRETEST FAILED` | Model/pair style/Kokkos problem on the first structure | Read `_pretest/pretest.screen` and `.stdout`; the script prints a hint |
| `MD FAILED: …` for one system | LAMMPS error in that system | `md.screen` (LAMMPS error) and `md.stdout` (Python traceback) in its folder |
| `equiv … FAIL` | ASE and LAMMPS differ by more than 1e-3 eV/Å | Check the head, the dtype and that both files come from the same checkpoint |
| Pilot `FAILED` | No site class worked, or the pilot run failed | `prep.log`, `pilot/pilot.screen`, `pilot/pilot.stdout` |
| All systems run the same stale structures | `SKIP_BUILD=1` reuses the last manifest | Rebuild, or edit/replace `manifest.txt` |
| `N structures to test` is larger than expected | Old defaults for sizes/coverages/heads/seeds | Print the `building structures:` line in `pipeline.log` and check the variables |

## Limits

- It tests **plain MD** and an **unbiased** pilot only. Production metadynamics is run by hand (see the commands in the project notes); a campaign script is not part of this file.
- Temperatures, contact counts and the Pt coordination after a few ps reflect the starting structure; use long runs (50–100 ps and more) before comparing conditions.
- A passing equivalence test shows LAMMPS = ASE, not that the model is accurate; check against DFT separately.
