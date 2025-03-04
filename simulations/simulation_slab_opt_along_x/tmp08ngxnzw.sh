#!/bin/bash
#BSUB -J simulation_slab_opt_along_x
#BSUB -q fotonano
#BSUB -n 32
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[block=16]"
#BSUB -oo simulation_slab_opt_along_x.out
#BSUB -eo simulation_slab_opt_along_x.err

source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && mpirun -np $LSB_DJOB_NUMPROC mpb-mpi r1=0.12 r2=0.2456985640859301 simulation_slab_opt_along_x.ctl