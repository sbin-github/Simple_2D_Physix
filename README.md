# Simple_2D_Physix

The codes should be run in each local directory that containing the code file!

Please generate 'model' and 'npy' folders in the following directory: 
'code/taichi/sand_collapse/solvers/data'

## Generate momentum npy data
Need to generate dataset for training and test from physical simulation.

$cd code/taichi/sand_collapse/solvers

$python3 dry_sand_collapse_quadratic_MLS_MPM_plane_strain.py

## Generate torch tensor dataset from npy data
'code/taichi/sand_collapse/solvers/learning/generate_pt_dataset.ipynb'

## Main training
'code/taichi/sand_collapse/solvers/learning/Stepwise_with_Disc.ipynb'
