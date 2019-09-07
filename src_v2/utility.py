import matplotlib.pyplot as plt
import numpy as np
import signal
import sys

def find_the_distance(mutated_input, last_node):
    # find root node
    root = last_node
    while(root.parent != None):
        root = root.parent

    # get the initial input from the root node
    initial_input = root.state.mutated_input

    # calc distance
    dist = np.sum((mutated_input - initial_input)**2) / mutated_input.size
    #print("dist", dist)
    return dist

figure_count = 1
def init_image_plots(rows, columns, image_size, figsize=(8, 8)):
    global figure_count
    plt.ion()
    figure_count += 1
    fig=plt.figure(figure_count, figsize=figsize)
    fig_plots = []
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        subplot = plt.imshow(np.random.randint(0,256,size=image_size))
        fig_plots.append(subplot)
    plt.show()
    return fig, fig_plots

def activate_ctrl_c_exit():
    def signal_handler(sig, frame):
            sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)