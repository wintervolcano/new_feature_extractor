#!/bin/bash
#SBATCH --job-name=ar_pro      # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=1             # CPUs per task
#SBATCH --time=02:00:00               # Maximum runtime
#SBATCH --mem=4G                      # Memory per node
#SBATCH --partition=short.q           # Partition/queue to use

file=$1 #archive file path
tag=$2 #save path + tag example /directory/test so the saved file would be /directory/test.npy

sing_img_ar="/hercules/:/hercules/ /hercules/scratch/vishnu/singularity_images/pulsarx_latest.sif"
sing_img="/hercules/:/hercules/ /hercules/scratch/vishnu/singularity_images/presto5_pddot.sif"
singularity exec -H /u/dbhatnagar:/home1 -B ~/.Xauthority -B $sing_img python3 pfd_processor.py --file "$file" --tag $tag --debug