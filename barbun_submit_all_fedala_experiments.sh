#!/bin/bash

# Batch submission script for FedALA regularization experiments
# This script submits a separate SLURM job for each configuration

USER="esunar"

# Define datasets and their corresponding num_clients
# datasets=(MUTAG BZR COX2 ENZYMES DD PROTEINS COLLAB IMDB-BINARY IMDB-MULTI)
num_clients=(5, 10, 5, 10, 10, 10, 10, 10, 10)

# Define regularization types
reg_types=(laplacian dirichlet)
# reg_types=(dirichlet)

# Define lambda values
lambdas=(0 0.5 1 1000)
# lambdas=(0 0.5 1)

echo "========================================================================"
echo "Submitting FedALA regularization experiments as separate SLURM jobs"
echo "========================================================================"
echo ""

# Counter for total jobs
total_jobs=0

# Loop through all combinations
for idx in "${!datasets[@]}"; do
    dataset="${datasets[$idx]}"
    clients="${num_clients[$idx]}"
    
    for reg_type in "${reg_types[@]}"; do
        for lambda in "${lambdas[@]}"; do
            # Create a descriptive job name
            JOB_NAME="fedala_${dataset}_${reg_type}_${lambda}"
            
            # Create experiment name for the config
            EXP_NAME="FedALA_${dataset}_${reg_type}_lambda${lambda}"
            
            echo "Submitting job: $JOB_NAME"
            echo "  Dataset: $dataset"
            echo "  Clients: $clients"
            echo "  Reg Type: $reg_type"
            echo "  Lambda: $lambda"
            echo "  Exp Name: $EXP_NAME"
            
            # Submit the job and capture job ID
            JOB_ID=$(sbatch --job-name="$JOB_NAME" \
                            --export=DATASET="$dataset",NUM_CLIENTS="$clients",REG_TYPE="$reg_type",LAMBDA="$lambda",EXP_NAME="$EXP_NAME" \
                            barbun_run_single_fedala.sh | awk '{print $4}')
            
            echo "  → Job ID: $JOB_ID"
            echo ""
            
            ((total_jobs++))
            
            # Small delay to avoid overwhelming the scheduler
            sleep 0.5
        done
    done
done

echo "========================================================================"
echo "All $total_jobs jobs submitted!"
echo "========================================================================"
echo ""
echo "Check job status with: squeue -u $USER"
echo "Monitor outputs in: slurm_job_outputs_barbun_2/"