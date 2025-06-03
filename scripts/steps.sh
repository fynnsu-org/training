#!/bin/bash

set -e

export WANDB_ENTITY=${WANDB_ENTITY:-"fschmitt-red-hat"}
export WANDB_PROJECT=${WANDB_PROJECT:-"cicd_demo"}

python ./scripts/create_wandb_run.py 1.5 "main"
python ./scripts/create_wandb_run.py 1.3

python ./scripts/create_wandb_report.py
