#!/bin/bash
#SBATCH -A esunar
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -p barbun
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_job_outputs_barbun_4/%j-job.out

module load apps/truba-ai/gpu-2024.0
module load lib/cuda/12.4
module load miniconda3
echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"

USER="esunar"

################################################################################

# Activate your conda environment
conda activate /arf/home/esunar/miniconda3/envs/openfgl

echo ""
echo "======================================================================================"
env
echo "======================================================================================"
echo ""

echo "======================================================================================"
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Python script..."
echo "==============================================================================="
echo "Experiment Name: ${EXP_NAME}"
echo "Dataset: ${DATASET}"
echo "Num Clients: ${NUM_CLIENTS}"
echo "Regularization Type: ${REG_TYPE}"
echo "Lambda: ${LAMBDA}"
echo "==============================================================================="

# Pass dataset as a single-element array and override experiment name
COMMAND="python -u run_experiments.py \
    --config configs/fedala_regularization_configs/bzr_fedala_baseline_clean.yaml \
    --dataset ${DATASET} \
    --num-clients ${NUM_CLIENTS} \
    --graph-reg-type ${REG_TYPE} \
    --lambda-graph ${LAMBDA} \
    --exp-name ${EXP_NAME} \
    --seeds 42 123 456"

echo ${COMMAND}
echo "-------------------------------------------"
$COMMAND

RET=$?
echo
echo "Solver exited with return code: $RET"
exit $RET