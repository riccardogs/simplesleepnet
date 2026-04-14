#!/bin/bash

# Define the experiment folder containing configuration files
EXPERIMENT_FOLDER="configs/across_seeds_"  # Modify this to the target folder
PROJECT_DIR="/cluster/home/shagupta/SimpleSleepNet"  # Project directory

# Slurm settings
SBATCH_SCRIPT="${PROJECT_DIR}/run_single_experiment.slurm"

# Check if the folder exists
if [ ! -d "${PROJECT_DIR}/${EXPERIMENT_FOLDER}" ]; then
    echo "Experiment folder ${PROJECT_DIR}/${EXPERIMENT_FOLDER} does not exist."
    exit 1
fi

# Iterate over all JSON config files in the folder
for config_file in "${PROJECT_DIR}/${EXPERIMENT_FOLDER}"/*.json; do
    # Check if the glob matches any files
    if [ -f "${config_file}" ]; then
        config_name=$(basename "${config_file}" .json)
        echo "Submitting job for configuration: ${config_name}"

        # Submit the Slurm job
        sbatch --export=CONFIG_FILE="${config_file}" "${SBATCH_SCRIPT}"
    else
        echo "No configuration files found in ${EXPERIMENT_FOLDER}."
    fi
done
