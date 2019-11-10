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


mkdir -p experiments/cifar/tensorfuzz

models=( "CIFAR_CNN" )
covs=( "snac" )

for model in "${models[@]}"
do
    for cov in "${covs[@]}"
    do
        echo "cifar - tensorfuzz - $model - $cov experiments"
        for (( counter=1; counter<=3; counter++ )) 
        do
            echo "cifar - tensorfuzz - $model - $cov iteration: $counter"
            python3 run_experiment.py --dataset CIFAR10 --params_set cifar10 $model tensorfuzz $cov --model $model --coverage $cov --runner tensorfuzz --random_seed $counter --image_verbose False > "experiments/cifar/tensorfuzz/$model-$cov-$counter.txt" 
        done
    done
done
