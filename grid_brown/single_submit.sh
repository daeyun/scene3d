#!/bin/bash
# Bash script for sumbitting matlab jobs to an SGE cluster queue.

# Parameters
NGT=1;

# Schedule Job
qsub -t 1:$NGT -l "inf gpus=1" single_driver.sh

# Use for benchmarking
# qsub -pe smp 2 -q '*@@ang' -l arch=lx26-amd64 

# Option explanations:
#  Mail Option (-m bea):  trigger email alerts at beginning, end, and aborting of jobs
#  Arch Option: (-l arch=...):  make sure 64 bit machines only
#  Batch option: (-t 1-3): submit array of jobs, each numbered 1,2, or 3

