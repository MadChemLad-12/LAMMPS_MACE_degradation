#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <stage_number> [nprod] [dt_ps]" >&2
  echo "Example: $0 3 200000 0.0005" >&2
  exit 1
fi

stage="$1"
nprod="${2:-200000}"   # default: 200,000 steps
dt="${3:-0.0005}"      # default: 0.0005 ps (0.5 fs), matching ptgr_naf.lmp
prev=$((stage - 1))

tag="md${stage}"
prev_tag="md${prev}"

# Resolve ROOT: use env $ROOT if set, else git top-level, else fallback path
if [[ -z "${ROOT:-}" ]]; then
  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    ROOT="$(git rev-parse --show-toplevel)"
  else
    ROOT="/home/h.jack/MACE_train/LAMMPS_MACE_degradation"
  fi
fi

lmp_in="${ROOT}/lammps/ptgr_naf/ptgr_naf.lmp"
model="${ROOT}/mace_V11_active_learning.model-mliap_lammps.pt"
pairfile="${ROOT}/lammps/ptgr_naf/pair_mace.lmp"

# Calculate simulated duration (ps and fs) using awk for floating-point math
read -r sim_ps sim_fs <<EOF
$(awk -v steps="$nprod" -v dt="$dt" 'BEGIN {
  ps = steps * dt;
  fs = ps * 1000.0;
  printf "%.4f %.2f\n", ps, fs;
}')
EOF

# Pick restart file from previous stage (preferring _final.restart, then _equil.restart, then newer rolling restart)
restartfile=""
if [[ "$stage" -gt 1 ]]; then
  if [[ -f "${prev_tag}_final.restart" ]]; then
    restartfile="${prev_tag}_final.restart"
  elif [[ -f "${prev_tag}_equil.restart" ]]; then
    restartfile="${prev_tag}_equil.restart"
  else
    latest_rolling="$(ls -t "${prev_tag}".*.restart 2>/dev/null | head -n 1 || true)"
    if [[ -n "$latest_rolling" ]]; then
      restartfile="$latest_rolling"
    fi
  fi
fi

echo "=== Simulation Summary (${tag}) ==="
echo "Steps (nprod) : ${nprod}"
echo "Timestep (dt) : ${dt} ps"
echo "Duration      : ${nprod} * ${dt} ps = ${sim_ps} ps (${sim_fs} fs)"

if [[ -n "$restartfile" ]]; then
  echo "Mode          : restart from ${restartfile}"
  echo "================================="
  lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -in "$lmp_in" -log "${tag}.log" -screen "${tag}.screen" \
    -var data md_equil.data -var userestart 1 -var restartfile "$restartfile" \
    -var model "$model" -var pairfile "$pairfile" \
    -var nmin 0 -var nheat 0 -var neq 0 -var nprod "$nprod" -var dt "$dt" -var useplumed 0 \
    -var tag "$tag" -var extxyz 1 -var ndump 200
else
  echo "Mode          : fresh start from md_equil.data (no ${prev_tag} restart found)"
  echo "================================="
  lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -in "$lmp_in" -log "${tag}.log" -screen "${tag}.screen" \
    -var data md_equil.data -var userestart 0 -var restartfile none.restart \
    -var model "$model" -var pairfile "$pairfile" \
    -var nmin 0 -var nheat 0 -var neq 0 -var nprod "$nprod" -var dt "$dt" -var useplumed 0 \
    -var tag "$tag" -var extxyz 1 -var ndump 200
fi