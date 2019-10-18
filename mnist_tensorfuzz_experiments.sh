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


rm -rf experiments/mnist/tensorfuzz
mkdir -p experiments/mnist/tensorfuzz
echo "cleaned experiments and created experiments directory"

models=( "LeNet1" "LeNet4" "LeNet5" )
covs=( "neuron" "kmn" "nbc" "tfc" )

for model in "${models[@]}"
do
    for cov in "${covs[@]}"
    do
        echo "mnist - tensorfuzz - $model - $cov experiments"
        for (( counter=1; counter<=3; counter++ )) 
        do
            echo "mnist - tensorfuzz - $model - $cov iteration: $counter"
            python3 run_experiment.py --dataset MNIST --params_set mnist $model tensorfuzz $cov --model $model --coverage $cov --runner tensorfuzz --random_seed $counter --image_verbose False > "experiments/mnist/tensorfuzz/$model-$cov-$counter.txt" 
        done
    done
done
