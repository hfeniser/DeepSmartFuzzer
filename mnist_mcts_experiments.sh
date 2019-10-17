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


rm -rf experiments/mnist/mcts
mkdir -p experiments/mnist/mcts
echo "cleaned and created experiments/mnist/mcts"

models=( "LeNet1" "LeNet4" "LeNet5" )
covs=( "neuron" "kmn" "nbc" "tfc" )

for model in "${models[@]}"
do
    for cov in "${covs[@]}"
    do
        echo "mnist - mcts - $model - $cov experiments"
        for (( counter=1; counter<=3; counter++ )) 
        do
            echo "mnist - mcts - $model - $cov iteration: $counter"
            python run_experiment.py --params_set mnist $model mcts $cov --dataset MNIST --model $model --coverage $cov --input_chooser random --runner mcts --image_verbose False >> "experiments/mnist/mcts/$model-$cov-$counter.txt" 
        done
    done
done

rm -rf experiments/mnist/mcts_clustered
mkdir -p experiments/mnist/mcts_clustered
echo "cleaned and created experiments/mnist/mcts_clustered"
for model in "${models[@]}"
do
    for cov in "${covs[@]}"
    do
        echo "mnist - mcts_clustered - $model - $cov experiments"
        for (( counter=1; counter<=3; counter++ )) 
        do
            echo "mnist - mcts_clustered - $model - $cov iteration: $counter"
            python run_experiment.py --params_set mnist $model mcts $cov --dataset MNIST --model $model --coverage $cov --input_chooser clustered_random --runner mcts_clustered --random_seed $counter --image_verbose False > "experiments/mnist/mcts_clustered/$model-$cov-$counter.txt" 
        done
    done
done