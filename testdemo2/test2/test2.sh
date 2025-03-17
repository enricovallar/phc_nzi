#!/bin/bash
#BSUB -J test2
#BSUB -q fotonano
#BSUB -n 10
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo test2.out
#BSUB -eo test2.err

source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && mpirun -np $LSB_DJOB_NUMPROC mpb-mpi  test2.ctl