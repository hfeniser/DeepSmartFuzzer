import numpy as np
from src.mcts import MCTS_Node, run_mcts
from src.RLforDL import RLforDL, RLforDL_State, Reward_Status
import glob, os

def mcts(params, experiment):
    game = RLforDL(params, experiment)
    
    fileList = glob.glob('data/mcts*', recursive=True)
    for f in fileList:
        os.remove(f)

    for i in range(params.nb_iterations):
        test_input, test_label = experiment.input_chooser(batch_size=params.batch_size)
        print(params.nb_iterations, params.batch_size, test_input.shape)
        root_state = RLforDL_State(test_input, 0, game=game)
        root = MCTS_Node(root_state, game)
        run_mcts(root, params.tc1, params.tc2, verbose=params.verbose, image_verbose=params.image_verbose)
        best_coverage, best_input = game.get_stat()
        game.reset_stat()
        if best_coverage > 0:
            experiment.coverage.step(best_input, update_state=True)
            if params.save_batch:
                np.save("data/mcts_{}.npy".format(i+1), best_input)
            print("IMAGE %g SUCCEED" % (i))
            print("found coverage increase", best_coverage)
            print("found different input", np.any(best_input-test_input != 0))
            print("Current Total Coverage", experiment.coverage.get_current_coverage())
        else:
            print("IMAGE %g FAILED" % (i))