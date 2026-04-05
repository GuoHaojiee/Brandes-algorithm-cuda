#!/bin/bash
#BSUB -J brandes_test
#BSUB -W 00:30
#BSUB -n 2
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o brandes_%J.out
#BSUB -e brandes_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC

module load SpectrumMPI

mpiexec -n 2 ./solution_mpi -in rmat-12
./validation -ans rmat-12.ans -res rmat-12.res
