#!/bin/bash
#SBATCH --job-name=fits_gen      # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=1             # CPUs per task
#SBATCH --time=02:00:00               # Maximum runtime
#SBATCH --mem=4G                      # Memory per node
#SBATCH --partition=short.q           # Partition/queue to use

config=$1
ar_file=$2 #archive file path
output=$3 
#tag=$2 #save path + tag example /directory/test so the saved file would be /directory/test.npy

sing_img_ar="/hercules/:/hercules/ /hercules/scratch/vishnu/singularity_images/pulsarx_latest.sif"
sing_img="/hercules/:/hercules/ /hercules/u/dbhatnagar/MAGIC/Ult_FE/presto5_pddot.sif"
#singularity exec -H /u/dbhatnagar:/home1 -B ~/.Xauthority -B $sing_img python3 FITS_generator_for_pfd.py --config "$config" --ar $ar_file --output $output
singularity exec -H /u/dbhatnagar:/home1 -B ~/.Xauthority -B $sing_img_ar python3 /hercules/u/dbhatnagar/MAGIC/Ult_FE/FITS_generator.py --config "$config" --ar $ar_file --output $output

#sbatch launch_fits_gen.sh /hercules/u/dbhatnagar/MAGIC/Ult_FE/features.json /hercules/scratch/dbhatnagar/UltFE_tests/test_arFiles/671711_59305.5645879304_cfbf00376_00001.ar /hercules/scratch/dbhatnagar/UltFE_tests/output_test/test1