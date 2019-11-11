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

mkdir -p experiments/adversarial/mcts

models=( "LeNet1")
covs=( "kmn" "nbc" "snac" "tfc" )

for model in "${models[@]}"
do
    for cov in "${covs[@]}"
    do
        echo "mnist - mcts - $model - $cov experiments"
        for (( counter=1; counter<=3; counter++ )) 
        do
            echo "mnist - mcts - $model - $cov iteration: $counter"
            python3 run_experiment.py --params_set mnist $model mcts $cov --dataset MNIST --model $model --coverage $cov --input_chooser random --runner mcts --random_seed $counter --image_verbose False --check_adversarial True > "experiments/adversarial/mcts/$model-$cov-$counter.txt" 
        done
    done
done

