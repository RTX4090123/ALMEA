#!/bin/bash

echo "Starting FBDB15K experiments"
bash run_experiments.sh 0 FBDB15K 500 0.2 0.45 0.35
#bash run_experiments.sh 0 FBDB15K 500 0.5 0.45 0.45
#bash run_experiments.sh 0 FBDB15K 500 0.8 0.45 0.95

echo "Starting FBYG15K experiments"
#bash run_experiments.sh 0 FBYG15K 300 0.2 0.50 0.25
#bash run_experiments.sh 0 FBYG15K 300 0.5 0.50 0.35
#bash run_experiments.sh 0 FBYG15K 300 0.8 0.50 0.45
echo "All experiments completed."
