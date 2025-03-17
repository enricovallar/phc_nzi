#!/bin/bash
#BSUB -J test_simple_lattices_slab
#BSUB -q fotonano
#BSUB -n 20
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo test_simple_lattices_slab.out
#BSUB -eo test_simple_lattices_slab.err

source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && mpirun -np $LSB_DJOB_NUMPROC mpb-mpi r1=0.2 h=0.4 test_simple_lattices_slab.ctl