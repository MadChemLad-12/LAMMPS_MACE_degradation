# 0. model equivalence (ASE vs LAMMPS)
python analysis/check_equiv.py --data data.lammps --ase-model MODEL.model --lmp-model model-mliap_lammps.pt \
  --lmp "lmp -k on g 1 -sf kk -pk kokkos newton on neigh half" --device cuda

# 1. plain MD: speed + stability (~15 ps)
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in ptgr_naf.lmp -log test.log \
  -var tag test -var model model-mliap_lammps.pt -var nmin 2000 -var nheat 10000 \
  -var neq 20000 -var nprod 0 -var useplumed 0 -var ndump 200
python analyze_run.py log test.log --nprod 4000000 --out test
python analyze_run.py traj test.extxyz --out test

# 2. unbiased PLUMED pilot on the equilibrated snapshot
python prep_plumed.py test_equil.data --list-sites
python prep_plumed.py test_equil.data --site vertex --mode unbiased --outdir pilot
cd pilot && lmp ... -in ../ptgr_naf.lmp -log pilot.log -var tag pilot -var userestart 1 \
  -var restartfile ../test_equil.restart -var nmin 0 -var nheat 0 -var neq 0 -var nprod 100000
python ../analyze_run.py colvar COLVAR --skip-ps 2 --out pilot   # prints SIGMA advice