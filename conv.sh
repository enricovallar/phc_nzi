#!/bin/sh
### General options
#BSUB -J meep_metasurface
#BSUB -R "span[hosts=1]"      # Ensure all cores are on the exact same node
#BSUB -R "rusage[mem=1GB]"    # Request 1GB of memory PER CORE
#BSUB -W 04:00                # Set walltime limit (hours:minutes)
#BSUB -o output_%J.out        # Standard output log
#BSUB -e error_%J.err         # Standard error log

### 1. Initialize conda for this non-interactive shell
source /zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh

### 2. Activate the environment
conda activate mpb-nzi-env

### 3. Run the script with MPI to actually use the 8 cores!
/zhome/2f/7/202918/miniconda3/envs/mpb-nzi-env/bin/mpirun -np $LSB_DJOB_NUMPROC python meep_convergence_test.py