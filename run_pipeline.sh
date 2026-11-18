#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh - build Pt/graphene/Nafion/water systems and smoke-test each one with the
# MACE model in LAMMPS:  build -> ASE-vs-LAMMPS equivalence -> short MD (min, heat, equilibrate)
# -> analysis -> (optional) unbiased PLUMED pilot -> summary table.
#
# Put it in the repo root (next to the mace_V9_... model files). Usage:
#   ./run_pipeline.sh                                  # defaults below
#   SIZES=0.8,1.0 COVERAGES=0.0,0.375 GPUS="0 1" ./run_pipeline.sh
#   ./run_pipeline.sh --dry-run                        # no GPU/MACE: soft dummy potential, checks the plumbing
#   ./run_pipeline.sh --skip-build --pilot             # reuse structures, add the PLUMED pilot
#   ./run_pipeline.sh -h
#
# Every setting is an environment variable (defaults in the CONFIG block). Outputs go to
# $OUT (default pipeline_runs/<timestamp>/): one folder per system with equiv.log, md.log,
# md.extxyz, md.lammpstrj, md_*png/csv, status.txt; plus summary.tsv.
# =============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ---------------- CONFIG ------------------------------------------------------
PYTHON="${PYTHON:-python}"
STRUCT_DIR="${STRUCT_DIR:-$ROOT/structure_gen}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT/analysis}"
LMP_DIR="${LMP_DIR:-$ROOT/lammps/ptgr_naf}"
LMP_INPUT="${LMP_INPUT:-$LMP_DIR/ptgr_naf.lmp}"
MODEL_ASE="${MODEL_ASE:-$ROOT/mace_V9_active_learning_stagetwo.model}"
MODEL_LMP="${MODEL_LMP:-$ROOT/mace_V9_active_learning_stagetwo.model-mliap_lammps.pt}"
HEAD="${HEAD:-}"                                   # head name for multihead models (check_equiv --head)
# structures
SIZES="${SIZES:-0.8,1.2}"             # nm, volume-equivalent diameters
COVERAGES="${COVERAGES:-0.25}"                # ML (per exposed surface Pt); 0.0 = bare
SITES="${SITES:-mixed}"                            # fcc|hcp|hollow|4fold|bridge|ontop|mixed
SEEDS="${SEEDS:-426}"
HEADS="${HEADS:-pt}"                                 # Nafion head modes: graphene,pt,free (empty = Config default)
SETS="${SETS:-}"                                   # extra Config overrides, space separated: "n_nafion=2 margin=7.0"
STRUCT_OUT="${STRUCT_OUT:-$STRUCT_DIR/structures_sweep}"
# LAMMPS
LMP="${LMP:-lmp -k on g 1 -sf kk -pk kokkos newton on neigh half}"   # check against your build's ML-IAP/Kokkos docs
GPUS="${GPUS:-1}"                                  # space-separated GPU ids; systems run in parallel, one per GPU
DEVICE="${DEVICE:-cuda}"
NMIN="${NMIN:-2000}"; NHEAT="${NHEAT:-5000}"; NEQ="${NEQ:-10000}"
NDUMP="${NDUMP:-200}"; DT="${DT:-0.0005}"          # DT in ps; must match -var dt
EQUIV="${EQUIV:-first}"                            # first | all | none  (ASE-vs-LAMMPS check)
EXTXYZ="${EXTXYZ:-0}"                              # 0 if your LAMMPS has no `dump extxyz` (converted afterwards)
# pilot
PILOT="${PILOT:-0}"; PILOT_STEPS="${PILOT_STEPS:-20000}"; PILOT_SITES="${PILOT_SITES:-vertex edge facet perimeter}"
DRY_RUN="${DRY_RUN:-0}"; SKIP_BUILD="${SKIP_BUILD:-0}"
OUT="${OUT:-$ROOT/pipeline_runs/$(date +%Y%m%d_%H%M%S)}"
# -------------------------------------------------------------------------------

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --pilot) PILOT=1 ;;
    -h|--help) usage ;;
    *) echo "unknown option $arg (-h for help)"; exit 2 ;;
  esac
done

PAIRFILE="$LMP_DIR/pair_mace.lmp"
EXTRA_VARS=""
if [ "$DRY_RUN" = 1 ]; then
  # soft dummy potential, low T and small dt so it cannot explode: tests files/walls/dumps/PLUMED only
  PAIRFILE="$LMP_DIR/pair_lj_test.lmp"; EQUIV=none
  [ "$LMP" = "lmp -k on g 1 -sf kk -pk kokkos newton on neigh half" ] && LMP="lmp"
  DT=0.0002; EXTRA_VARS="-var tstart 10 -var temp 50 -var dt 0.0002"
  echo "DRY RUN: dummy potential, no MACE, no equivalence test. Results say nothing about the model."
fi

mkdir -p "$OUT"
LOG="$OUT/pipeline.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------------- preflight ------------------------------------------------------
fail_pre=0
need() { command -v "$1" >/dev/null 2>&1 || { say "MISSING command: $1"; fail_pre=1; }; }
need "$PYTHON"; need "$(echo "$LMP" | awk '{print $1}')"
[ "$SKIP_BUILD" = 1 ] || need packmol
[ "$PILOT" = 1 ] && [ "$DRY_RUN" = 0 ] && need plumed
for f in "$STRUCT_DIR/sweep_build.py" "$STRUCT_DIR/build_interface.py" "$STRUCT_DIR/prep_plumed.py" \
         "$ANALYSIS_DIR/analyze_run.py" "$ANALYSIS_DIR/check_equiv.py" "$ANALYSIS_DIR/summarize_pipeline.py" \
         "$LMP_INPUT" "$PAIRFILE"; do
  [ -f "$f" ] || { say "MISSING file: $f"; fail_pre=1; }
done
if [ "$DRY_RUN" = 0 ]; then
  for f in "$MODEL_ASE" "$MODEL_LMP"; do [ -f "$f" ] || { say "MISSING model: $f (dvc pull?)"; fail_pre=1; }; done
fi
"$PYTHON" - <<'PY' || fail_pre=1
import importlib.util, sys
bad = [m for m in ("numpy", "scipy", "ase", "matplotlib") if importlib.util.find_spec(m) is None]
if bad:
    print("MISSING python modules:", bad); sys.exit(1)
PY
[ "$fail_pre" = 0 ] || { say "preflight failed - fix the above and re-run"; exit 2; }
say "outputs: $OUT"

# ---------------- 1. build ----------------------------------------------------------
if [ "$SKIP_BUILD" = 1 ]; then
  say "skipping build; using manifest in $STRUCT_OUT"
else
  say "building structures: sizes=$SIZES coverages=$COVERAGES sites=$SITES seeds=$SEEDS heads=${HEADS:-<Config default>}"
  SET_ARGS=""; for s in $SETS; do SET_ARGS="$SET_ARGS --set $s"; done
  HEAD_ARG=""; [ -n "$HEADS" ] && HEAD_ARG="--heads $HEADS"
  ( cd "$STRUCT_DIR" && "$PYTHON" sweep_build.py --out "$STRUCT_OUT" --sizes "$SIZES" --coverages "$COVERAGES" \
      --sites "$SITES" --seeds "$SEEDS" $HEAD_ARG $SET_ARGS ) 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { say "structure build FAILED"; exit 1; }
fi
MANIFEST="$STRUCT_OUT/manifest.txt"
[ -s "$MANIFEST" ] || { say "no structures in $MANIFEST"; exit 1; }
mapfile -t DATAFILES < "$MANIFEST"
say "${#DATAFILES[@]} structures to test"

# ---------------- per-system test ------------------------------------------------------
run_one() {   # run_one <data.lammps> <gpu> <index>
  local data="$1" gpu="$2" idx="$3"
  local tag cond id dir
  tag="$(basename "$(dirname "$data")")"; cond="$(basename "$(dirname "$(dirname "$data")")")"
  id="${cond}__${tag}"; dir="$OUT/$id"; mkdir -p "$dir"; cd "$dir" || return 1
  local natoms; natoms=$(awk '/atoms$/ {print $1; exit}' "$data")
  echo "atoms=$natoms build=ok" > status.txt
  export CUDA_VISIBLE_DEVICES="$gpu"
  local note=""

  # 2a. ASE vs LAMMPS equivalence
  if [ "$EQUIV" = all ] || { [ "$EQUIV" = first ] && [ "$idx" = 0 ]; }; then
    local hd=""; [ -n "$HEAD" ] && hd="--head $HEAD"
    "$PYTHON" "$ANALYSIS_DIR/check_equiv.py" --data "$data" --ase-model "$MODEL_ASE" --lmp-model "$MODEL_LMP" \
        --lmp "$LMP" --device "$DEVICE" $hd --keep equiv_files > equiv.log 2>&1
    grep -E "PASS|FAIL" equiv.log | tail -1 | sed 's/^/   equiv: /'
    grep -q "PASS" equiv.log || note="${note}equiv_failed;"
  fi

  # 2b. plain MD: minimise, heat, equilibrate (no PLUMED)
  $LMP -in "$LMP_INPUT" -log md.log -screen md.screen \
      -var data "$data" -var model "$MODEL_LMP" -var pairfile "$PAIRFILE" -var tag md \
      -var nmin "$NMIN" -var nheat "$NHEAT" -var neq "$NEQ" -var nprod 0 -var useplumed 0 \
      -var ndump "$NDUMP" -var extxyz "$EXTXYZ" $EXTRA_VARS > md.stdout 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || grep -q "^ERROR" md.screen 2>/dev/null; then
    echo "atoms=$natoms build=ok md=FAILED note=${note}see_md.screen" > status.txt
    echo "   MD FAILED: $(grep -m1 '^ERROR' md.screen md.stdout 2>/dev/null | head -1)"
    return 1
  fi
  # Fallback conversion to md.extxyz if native extxyz dump is disabled (EXTXYZ=0)
  if [ ! -f md.extxyz ]; then
    "$PYTHON" - << 'PY' > convert.log 2>&1
import numpy as np
from ase.io import read, write

try:
    # Read custom LAMMPS trajectory (adjust filename if your script uses a different extension)
    images = read("md.lammpstrj", index=":")
    
    # If you also printed potential energy via fix print to md_pe.dat, attach it:
    if np.path.exists("md_pe.dat"):
        pe_data = np.loadtxt("md_pe.dat", comments="#")
        for atoms, (_, pe) in zip(images, pe_data):
            atoms.info["energy"] = float(pe)
            
    write("md.extxyz", images, format="extxyz")
except Exception as e:
    print(f"Conversion failed: {e}")
    exit(1)
PY
  fi

  [ -f md.extxyz ] || { [ -f md.lammpstrj ] && "$PYTHON" "$ANALYSIS_DIR/analyze_run.py" convert md.lammpstrj --out md.extxyz > convert.log 2>&1; }

  # 2c. analysis
  "$PYTHON" "$ANALYSIS_DIR/analyze_run.py" log md.log --nprod 4000000 --out md > analyze_log.txt 2>&1
  "$PYTHON" "$ANALYSIS_DIR/analyze_run.py" traj md.extxyz --out md --dt "$DT" > analyze_traj.txt 2>&1 \
      || note="${note}traj_analysis_failed;"
  local pilot=""

  # 2d. optional unbiased PLUMED pilot from the equilibrated restart
  if [ "$PILOT" = 1 ]; then
    pilot="FAILED"
    for site in $PILOT_SITES; do
      if "$PYTHON" "$STRUCT_DIR/prep_plumed.py" md_equil.data --site "$site" --mode unbiased --outdir pilot \
           --stride 20 > prep.log 2>&1; then
        ( cd pilot && $LMP -in "$LMP_INPUT" -log pilot.log -screen pilot.screen \
            -var data "$data" -var model "$MODEL_LMP" -var pairfile "$PAIRFILE" -var tag pilot \
            -var userestart 1 -var restartfile ../md_equil.restart -var nmin 0 -var nheat 0 -var neq 0 \
            -var nprod "$PILOT_STEPS" -var plumed plumed.dat -var useplumed 1 -var ndump "$NDUMP" \
            -var extxyz "$EXTXYZ" $EXTRA_VARS > pilot.stdout 2>&1 ) \
          && [ -s pilot/COLVAR ] && {
            "$PYTHON" "$ANALYSIS_DIR/analyze_run.py" colvar pilot/COLVAR --out pilot/pilot > pilot/analyze_colvar.txt 2>&1
            pilot="ok($site)"; break; }
      fi
    done
    [ "$pilot" = FAILED ] && note="${note}pilot_failed;"
  fi
  echo "atoms=$natoms build=ok md=ok pilot=${pilot:--} note=${note:--}" > status.txt
  echo "   done: $(grep -m1 'last run speed' analyze_log.txt)"
  return 0
}

# ---------------- run all systems, one per GPU at a time --------------------------------
read -r -a GPU_LIST <<< "$GPUS"
NG=${#GPU_LIST[@]}
say "running ${#DATAFILES[@]} systems on GPU(s): ${GPU_LIST[*]}  (dry-run=$DRY_RUN, equivalence=$EQUIV, pilot=$PILOT)"
i=0
for data in "${DATAFILES[@]}"; do
  gpu="${GPU_LIST[$((i % NG))]}"
  say ">> $(basename "$(dirname "$(dirname "$data")")")/$(basename "$(dirname "$data")")  (gpu $gpu)"
  ( run_one "$data" "$gpu" "$i" ) >> "$LOG" 2>&1 &
  i=$((i + 1))
  if [ $((i % NG)) -eq 0 ]; then wait; fi
done
wait

# ---------------- summary ----------------------------------------------------------------------
say "summary"
"$PYTHON" "$ANALYSIS_DIR/summarize_pipeline.py" "$OUT" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
