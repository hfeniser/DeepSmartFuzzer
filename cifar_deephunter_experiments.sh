#!/bin/bash

# this function is called when Ctrl-C is sent
function trap_ctrlc ()
{
    echo " Stopping"
    exit 2
}
# initialise trap to call trap_ctrlc function
# when signal 2 (SIGINT) is received
trap "trap_ctrlc" 2

mkdir -p experiments/cifar/deephunter

models=( "CIFAR_CNN" )
covs=( "snac" )

for model in "${models[@]}"
do
    for cov in "${covs[@]}"
    do
        echo "cifar - deephunter - $model - $cov experiments"
        for (( counter=1; counter<=3; counter++ )) 
        do
            echo "cifar - deephunter - $model - $cov iteration: $counter"
            python3 run_experiment.py --params_set cifar10 $model deephunter $cov --dataset CIFAR10 --model $model --coverage $cov --runner deephunter --random_seed $counter --image_verbose False > "experiments/cifar/deephunter/$model-$cov-$counter.txt" 
        done
    done
done
