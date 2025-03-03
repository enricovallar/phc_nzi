#!/bin/bash
#BSUB -J test_changes_slab
#BSUB -q fotonano
#BSUB -n 20
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo test_changes_slab.out
#BSUB -eo test_changes_slab.err

source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && mpirun -np $LSB_DJOB_NUMPROC mpb-mpi r1=0.1 r2=0.339 test_changes_slab.ctl