#!/bin/bash
#BSUB -J simulation_InP_optimization
#BSUB -q fotonano
#BSUB -n 32
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[block=8]"
#BSUB -oo simulation_InP_optimization.out
#BSUB -eo simulation_InP_optimization.err
#BSUB -u s232699@dtu.dk
#BSUB -oo simulation_InP_optimization/simulation_InP_optimization.out
#BSUB -eo simulation_InP_optimization/simulation_InP_optimization.err
module purge
source /zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh
conda activate nzi-mp
mpirun -np 32 python /zhome/2f/7/202918/phc_nzi/sources/mpi_differential_evolution.py --run_opt --param_names="r1,r2" --simulation_name="simulation_InP_optimization" --maxiter=20 --polarization="tm" --param_bounds 0.05,0.15 0.2,0.4 --popsize=16 --strategy="rand1bin"
