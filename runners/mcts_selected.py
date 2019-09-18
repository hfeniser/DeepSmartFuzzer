import numpy as np
from src.mcts import MCTS_Node, run_mcts
from src.RLforDL import RLforDL, RLforDL_State, Reward_Status

def mcts_selected(params, experiment):
    game = RLforDL(experiment.coverage, params.input_shape, params.input_lower_limit, params.input_upper_limit,\
        params.action_division_p1, params.actions_p2, params.tc3, with_implicit_reward=params.implicit_reward)

    import glob, os

    fileList = glob.glob('data/mcts*', recursive=True)
    for f in fileList:
        os.remove(f)

    for i in range(params.nb_iterations):
        test_input = np.load("data/deephunter_{}.npy".format((i%10)+1))
        root_state = RLforDL_State(test_input, 0, game=game)
        root = MCTS_Node(root_state, game)
        run_mcts(root, params.tc1, params.tc2)
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