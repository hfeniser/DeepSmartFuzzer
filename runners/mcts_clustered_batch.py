import numpy as np
from src.mcts import MCTS_Node, run_mcts
from src.RLforDL import RLforDL, RLforDL_State, Reward_Status

def mcts_clustered_batch(params, experiment):
    if params.input_chooser != "clustered_random":
        raise Exception("Incompatible Runner:mcts_clustered_batch and Input Chooser:" + str(params.input_chooser))

    game = RLforDL(experiment.coverage, params.input_shape, params.input_lower_limit, params.input_upper_limit,\
        params.action_division_p1, params.actions_p2, params.tc3, with_implicit_reward=params.implicit_reward)

    mcts_roots = [None] * len(experiment.input_chooser)

    for i in range(0, 30):
        cluster_index, (test_input, _) = experiment.input_chooser(batch_size=64, cluster_index=1)
        
        if params.verbose:
            print("cluster_index", cluster_index)
        
        root_state = RLforDL_State(test_input, 0, game=game)
        if mcts_roots[cluster_index] == None:
            mcts_roots[cluster_index] = MCTS_Node(root_state, game)
        else:
            mcts_roots[cluster_index].updateRootWithNewInput(root_state)
        run_mcts(mcts_roots[cluster_index], params.tc1, params.tc2)
        
        best_coverage, best_input = game.get_stat()
        game.reset_stat()
        if best_coverage > 0:
            experiment.coverage.step(best_input, update_state=True)
            np.save("data/mcts_{}.npy".format(i+1), best_input)
            print("IMAGE %g SUCCEED" % (i))
            print("found coverage increase", best_coverage)
            print("found different input", np.any(best_input-test_input != 0))
            print("Current Total Coverage", experiment.coverage.get_current_coverage())
        else:
            print("IMAGE %g FAILED" % (i))