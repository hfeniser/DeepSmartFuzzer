
import parent_import # adds parent directory to sys.path

from src_v2.utility import init_image_plots
import numpy as np
import time

fig_current, fig_plots_current = init_image_plots(4,4, (28,28))

for i in range(30):
    try:
        mutated_input = np.load("data/mcts_{}.npy".format(i+1))
        fig_current.suptitle("Batch #"+str(i))
        for j in range(len(mutated_input[0:16])):
            fig_plots_current[j].set_data(mutated_input[j].reshape((28,28)))
        fig_current.canvas.flush_events()
        time.sleep(1)
    except:
        pass