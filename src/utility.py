import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
import argparse

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
def init_image_plots(rows, columns, input_shape, figsize=(8, 8)):
    global figure_count
    image_size = get_image_size(input_shape)
    plt.ion()
    figure_count += 1
    fig=plt.figure(figure_count, figsize=figsize)
    fig_plots = []
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        subplot = plt.imshow(np.random.randint(0,256,size=image_size))
        fig_plots.append(subplot)
    plt.show()
    return (fig, fig_plots)

def update_image_plots(f, images, title):
    (fig, fig_plots) = f
    if images.shape[-1] == 1:
        images = images.reshape(images.shape[:-1])
    fig.suptitle(title)
    for j in range(len(images)):
        fig_plots[j].set_data(images[j])
    fig.canvas.draw()
    fig.canvas.flush_events()

def activate_ctrl_c_exit():
    def signal_handler(sig, frame):
            sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def merge_object(initial_obj, additional_obj):
    for property in additional_obj.__dict__:
        setattr(initial_obj, property, getattr(additional_obj, property))

    return initial_obj

def get_image_size(input_shape):
    image_size = input_shape
    if len(input_shape) == 4:
        image_size = image_size[1:]
    if image_size[-1] == 1:
        image_size = image_size[:-1]
    return image_size