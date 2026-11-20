### NOTES FOR FUTURE SELF

AT the moment I am working on establishing a framework to use plummed and mace together. 

as of 11/20 I am running 
```
ROOT=~/MACE_train/LAMMPS_MACE_degradation
cd $ROOT/pipeline_runs/20261119_145955_prod
for d in cov*/; do ( cd "$d" && python $ROOT/analysis/analyze_run.py traj md.extxyz --out md --dt 0.0005 > analyze_traj.txt 2>&1 ); done
for d in cov*/; do ( cd "$d" && python $ROOT/analysis/analyze_run.py bonds md.extxyz --out md > analyze_bonds.txt 2>&1 ); done
python $ROOT/summarize_pipeline.py .

cd $ROOT/pipeline_runs/20261119_145955_prod/cov0.375_mixed_s426_graphene__octahedron_055pt_O0.391ML_s426
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -in $ROOT/lammps/ptgr_naf/ptgr_naf.lmp -log md2.log -screen md2.screen \
  -var data md_equil.data -var userestart 1 -var restartfile md_equil.restart \
  -var model $ROOT/mace_V11_active_learning.model-mliap_lammps.pt \
  -var pairfile $ROOT/lammps/ptgr_naf/pair_mace.lmp \
  -var nmin 0 -var nheat 0 -var neq 0 -var nprod 200000 -var useplumed 0 \
  -var tag md2 -var extxyz 1 -var ndump 200
python $ROOT/analysis/analyze_run.py traj md2.extxyz --out md2 --dt 0.0005

cd $ROOT/pipeline_runs/20261119_145955_prod/cov0.000_none_s426_pt__octahedron_019pt
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -in $ROOT/lammps/ptgr_naf/ptgr_naf.lmp -log md2.log -screen md2.screen \
  -var data md_equil.data -var userestart 1 -var restartfile md_equil.restart \
  -var model $ROOT/mace_V11_active_learning.model-mliap_lammps.pt \
  -var pairfile $ROOT/lammps/ptgr_naf/pair_mace.lmp \
  -var nmin 0 -var nheat 0 -var neq 0 -var nprod 200000 -var useplumed 0 \
  -var tag md2 -var extxyz 1 -var ndump 200
python $ROOT/analysis/analyze_run.py traj md2.extxyz --out md2 --dt 0.0005

cd /home/h.jack/MACE_train/LAMMPS_MACE_degradation/pipeline_runs/20261119_145955_prod/cov0.375_mixed_s426_graphene__octahedron_019pt_O0.333ML_s426
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -in $ROOT/lammps/ptgr_naf/ptgr_naf.lmp -log md2.log -screen md2.screen \
  -var data md_equil.data -var userestart 1 -var restartfile md_equil.restart \
  -var model $ROOT/mace_V11_active_learning.model-mliap_lammps.pt \
  -var pairfile $ROOT/lammps/ptgr_naf/pair_mace.lmp \
  -var nmin 0 -var nheat 0 -var neq 0 -var nprod 200000 -var useplumed 0 \
  -var tag md2 -var extxyz 1 -var ndump 200
python $ROOT/analysis/analyze_run.py traj md2.extxyz --out md2 --dt 0.0005

cd /home/h.jack/MACE_train/LAMMPS_MACE_degradation/pipeline_runs/20261119_145955_prod/cov0.500_mixed_s426_graphene__octahedron_055pt_O0.522ML_s426
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -in $ROOT/lammps/ptgr_naf/ptgr_naf.lmp -log md2.log -screen md2.screen \
  -var data md_equil.data -var userestart 1 -var restartfile md_equil.restart \
  -var model $ROOT/mace_V11_active_learning.model-mliap_lammps.pt \
  -var pairfile $ROOT/lammps/ptgr_naf/pair_mace.lmp \
  -var nmin 0 -var nheat 0 -var neq 0 -var nprod 200000 -var useplumed 0 \
  -var tag md2 -var extxyz 1 -var ndump 200
python $ROOT/analysis/analyze_run.py traj md2.extxyz --out md2 --dt 0.0005
```
This will give me an analyusis of how it went. 

Once this is done I would like to also like to do a comparision with mace and dft maybe. I might also add an rdf function to bonds section of analyse run.py.