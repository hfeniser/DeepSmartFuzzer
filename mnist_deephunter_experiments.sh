#!/bin/bash

rm -rf experiments/mnist/deephunter
mkdir -p experiments/mnist/deephunter
echo "cleaned experiments and created experiments directory"

models=( "LeNet1" "LeNet4" "LeNet5" )
covs=( "neuron" "kmn" "nbc" "tfc" )

for model in "${models[@]}"
do
    for cov in "${covs[@]}"
    do
        echo "mnist - deephunter - $model - $cov experiments"
        for (( counter=1; counter<=3; counter++ )) 
        do
            echo "mnist - deephunter - $model - $cov iteration: $counter"
            python run_experiment.py --params_set deephunter_mnist --dataset MNIST --model $model --coverage $cov --runner deephunter --image_verbose False >> "experiments/mnist/deephunter/$model-$cov-$counter.txt" 
        done
    done
done