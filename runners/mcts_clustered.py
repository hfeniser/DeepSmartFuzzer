import numpy as np
from src.mcts import MCTS_Node, run_mcts
from src.DeepSmartFuzzer import DeepSmartFuzzer, DeepSmartFuzzer_State, Reward_Status
import glob, os, time

def mcts_clustered(params, experiment):
    if params.input_chooser != "clustered_random":
        raise Exception("Incompatible Runner:mcts_clustered_batch and Input Chooser:" + str(params.input_chooser))

    game = DeepSmartFuzzer(params, experiment)

    experiment.iteration = 0
    while not experiment.termination_condition():
        cluster_index, (test_input, test_label) = experiment.input_chooser(batch_size=params.batch_size)
        
        if params.verbose:
            print("cluster_index", cluster_index)
        
        root_state = DeepSmartFuzzer_State(test_input, 0, game=game)
        root = MCTS_Node(root_state, game)
        run_mcts(root, params.tc1, params.tc2, verbose=params.verbose, image_verbose=params.image_verbose)
        
        best_coverage, best_input = game.get_stat()
        game.reset_stat()
        if best_coverage > 0:
            experiment.coverage.step(best_input, update_state=True)
            experiment.input_chooser.append(best_input, test_label)
        if params.verbose:
            print("iteration: %g" % (experiment.iteration))
            print("found coverage increase", best_coverage)
            print("Current Total Coverage", experiment.coverage.get_current_coverage())
        
        experiment.iteration += 1