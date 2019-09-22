
from src.utility import init_image_plots, update_image_plots
import numpy as np
import time

fig = init_image_plots(4,4, (28,28))

for i in range(0,30):
    try:
        mutated_input = np.load("data/mcts_{}.npy".format(i+1))
    except Exception as e:
        print(e)
        continue

    update_image_plots(fig, mutated_input.reshape(-1,28,28), "Batch #"+str(i+1))
    time.sleep(2)