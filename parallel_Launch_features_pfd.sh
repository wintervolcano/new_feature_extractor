#!/bin/bash

# Directory containing input files
INPUT_DIR=$1
CONFIG=$2
OUTPUT_PATH=$3 #where output fits files will be stored
# Directory for output files
#OUTPUT_DIR=$2
# Path to the singularity script
SBATCH_SCRIPT="/hercules/u/dbhatnagar/MAGIC/Ult_FE/launch_fits_gen.sh"
# Maximum number of jobs allowed to run simultaneously
MAX_JOBS=100

# Ensure the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Find and sort all .ar files in the input directory
PFD_FILES=($(ls "$INPUT_DIR"/*.pfd 2>/dev/null | sort))

# Check if .ar files exist
if [ ${#PFD_FILES[@]} -eq 0 ]; then
    echo "No .ar files found in the directory: $INPUT_DIR"
    exit 1
fi

# Iterate over sorted .ar files
for INPUT_FILE in "${PFD_FILES[@]}"; do
    # Extract base name without extension
    BASENAME=$(basename "$INPUT_FILE" .ar)

    # Define the output path
    OUTPUT="${OUTPUT_PATH}/${BASENAME}"

    # Wait if max jobs are running
    while [ "$(squeue -u $USER | wc -l)" -ge "$MAX_JOBS" ]; do
        echo "Waiting for jobs to complete... Current jobs: $(squeue -u $USER | wc -l)"
        sleep 15
    done

    # Submit the job
    sbatch "$SBATCH_SCRIPT" "$CONFIG" "$INPUT_FILE" "$OUTPUT"
    echo "Submitted job for $INPUT_FILE with output $OUTPUT"
done