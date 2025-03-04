#!/bin/bash
#BSUB -J simulation_slab_unoptimized
#BSUB -q fotonano
#BSUB -n 32
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[block=8]"
#BSUB -oo simulation_slab_unoptimized.out
#BSUB -eo simulation_slab_unoptimized.err

source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && mpirun -np $LSB_DJOB_NUMPROC mpb-mpi r1=0.2 r2=0.3 simulation_slab_unoptimized.ctl