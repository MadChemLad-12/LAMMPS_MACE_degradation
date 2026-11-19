ROOT=~/MACE_train/LAMMPS_MACE_degradation
cd $ROOT/pipeline_runs/20261119_112224_short_trailrun/cov0.000_none_s426_graphene__octahedron_019pt

# A. read the pilot: CV fluctuations -> SIGMA advice
python $ROOT/analysis/analyze_run.py colvar pilot/COLVAR --skip-ps 2 --out pilot/pilot

# B. pick a target and write the metadynamics PLUMED file
python $ROOT/structure_gen/prep_plumed.py md_equil.data --list-sites
python $ROOT/structure_gen/prep_plumed.py md_equil.data --site vertex --mode metad \
    --sigma 0.3,0.3 --outdir metad_vertex          # replace with the SIGMA from step A

# C. run it
cd metad_vertex
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -in $ROOT/lammps/ptgr_naf/ptgr_naf.lmp -log metad.log -screen metad.screen \
  -var data ../md_equil.data -var userestart 1 -var restartfile ../md_equil.restart \
  -var model $ROOT/mace_V11_active_learning.model-mliap_lammps.pt \
  -var pairfile $ROOT/lammps/ptgr_naf/pair_mace.lmp \
  -var nmin 0 -var nheat 0 -var neq 0 -var nprod 100000 \
  -var plumed plumed.dat -var useplumed 1 -var tag metad -var extxyz 1 -var ndump 500

# D. analyse
python $ROOT/analysis/analyze_run.py log metad.log --out metad
python $ROOT/analysis/analyze_run.py colvar COLVAR --out metad
python $ROOT/analysis/analyze_run.py fes HILLS --bias-factor 15 --temp 353 --out metad
python $ROOT/analysis/analyze_run.py merge metad.extxyz COLVAR --out metad_cv.extxyz