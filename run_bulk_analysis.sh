#!/usr/bin/env bash
set -euo pipefail

round="2"

dirs=(
  "cov0.000_none_s426_pt__octahedron_019pt"
  "cov0.375_mixed_s426_graphene__octahedron_019pt_O0.333ML_s426"
  "cov0.375_mixed_s426_graphene__octahedron_055pt_O0.391ML_s426"
  "cov0.500_mixed_s426_graphene__octahedron_055pt_O0.522ML_s426"
)


pipruns_run="/home/h.jack/MACE_train/LAMMPS_MACE_degradation/pipeline_runs/20261119_145955_prod"
script="/home/h.jack/MACE_train/LAMMPS_MACE_degradation/analysis/analyze_run.py"

for dir in "${dirs[@]}"; do
  run_dir="$pipruns_run/$dir"
  if [[ ! -d "$run_dir" ]]; then
    echo "Skipping missing directory: $run_dir" >&2
    continue
  fi

  (
    cd "$run_dir"
    python "$script" log "md${round}.log" --out "md${round}" > "analyse_log${round}.log"
    python "$script" traj "md${round}.extxyz" --out "md${round}" --dt 0.0005 > "analyse_tra${round}.log"
    python "$script" bonds "md${round}.extxyz" --out "md${round}" > "analyse_bonds${round}.log"
  )
  echo "Finished $run_dir"
done