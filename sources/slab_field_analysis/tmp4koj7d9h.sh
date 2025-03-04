#!/bin/bash
#BSUB -J slab_field_analysis
#BSUB -q fotonano
#BSUB -n 25
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo slab_field_analysis.out
#BSUB -eo slab_field_analysis.err

source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && mpirun -np $LSB_DJOB_NUMPROC mpb-mpi r=0.35 slab_field_analysis.ctl