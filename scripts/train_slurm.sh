#!/usr/bin/env bash
# Template only; set your cluster's account/partition and environment before submit.
#SBATCH --job-name=prmshift
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=prmshift-%j.out
set -euo pipefail
: "${PRMSHIFT_PYTHON:?Set PRMSHIFT_PYTHON to your cluster environment python}"
: "${PRMSHIFT_REPO:?Set PRMSHIFT_REPO to the repository path}"
cd "$PRMSHIFT_REPO"
export PYTHONPATH="$PRMSHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"$PRMSHIFT_PYTHON" -m prmshift train --config "${PRMSHIFT_CONFIG:-configs/hf_1_5b.yaml}"
