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

echo "starting deephunter"
./mnist_deephunter_experiments.sh
echo "finishing deephunter"

echo "starting tensorfuzz"
./mnist_tensorfuzz_experiments.sh 
echo "finishing tensorfuzz"

echo "starting mcts"
./mnist_mcts_experiments.sh 
echo "finishing mcts"
