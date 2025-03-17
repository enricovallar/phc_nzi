#!/bin/bash
#BSUB -J test_optimization_slab
#BSUB -q fotonano
#BSUB -n 10
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo test_optimization_slab/test_optimization_slab.out
#BSUB -eo test_optimization_slab/test_optimization_slab.err
module purge
source /zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh
conda activate nzi-mp
mpirun -np 10 python /zhome/2f/7/202918/phc_nzi/src/mpi_differential_evolution.py --run_opt --param_names="r1,r2" --simulation_name="test_optimization_slab" --maxiter=15 --polarization="zodd" --param_bounds 0.05,0.15 0.2,0.4 --popsize=5 --strategy="rand1bin" --bands 2 3 4
