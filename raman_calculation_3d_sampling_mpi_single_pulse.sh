#!/bin/bash
#SBATCH -t 47:59:59
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=3
#SBATCH --partition=milano
#SBATCH --account=lcls:tmo100827624
#SBATCH --output=/sdf/data/lcls/ds/tmo/tmo100827624/results/erik/raman_calculation/logs/log_calcualtion_single_pulse.log

mpirun python raman_calculation_3d_sampling_mpi_single_pulse.py