import traceback
from os import path, makedirs
import h5py
import numpy as np
import sys
from cleverhans.attacks import SaliencyMapMethod, FastGradientMethod, CarliniWagnerL2, BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from keras.datasets import mnist, cifar10
from keras.models import model_from_json
from keras.utils import np_utils
from keras import models
from math import ceil
from sklearn.metrics import classification_report, confusion_matrix


def load_CIFAR(one_hot=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def load_MNIST(one_hot=True, channel_first=True):
    """
    Load MNIST data
    :param one_hot:
    :return:
    """
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess dataset
    # Normalization and reshaping of input.
    if channel_first:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        # For output, it is important to change number to one-hot vector.
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def load_model(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into model
    model.load_weights(model_name + '.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Model structure loaded from ", model_name)
    return model


def get_layer_outs_old(model, class_specific_test_set):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    # Testing
    layer_outs = [func([class_specific_test_set, 1.]) for func in functors]

    return layer_outs


def get_layer_outs(model, test_input, skip=[]):

    inp = model.input  # input placeholder
    outputs = [layer.output for index, layer in enumerate(model.layers) \
               if index not in skip]

    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layer_outs = [func([test_input]) for func in functors]

    return layer_outs


def get_layer_outs_new(model, test_input, skip=[]):
    evaluator = models.Model(inputs=model.input,
                             outputs=[layer.output for index, layer in enumerate(model.layers)
                                      if index not in skip][1:])

    return evaluator.predict(test_input) #np.expand_dims( test_input, axis=0))


def calc_major_func_regions(model, train_inputs, skip=None):
    if skip is None:
        skip = []

    outs = get_layer_outs_new(model, train_inputs, skip)

    major_regions = []

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        layer_out = layer_out.mean(axis=tuple(i for i in range(1, layer_out.ndim - 1)))

        major_regions.append((layer_out.min(axis=0), layer_out.max(axis=0)))

    return major_regions


def get_layer_outputs_by_layer_name(model, test_input, skip=None):
    if skip is None:
        skip = []

    inp = model.input  # input placeholder
    outputs = {layer.name: layer.output for index, layer in enumerate(model.layers)
               if (index not in skip and 'input' not in layer.name)}  # all layer outputs (except input for functionals)
    functors = {name: K.function([inp], [out]) for name, out in outputs.items()}  # evaluation functions

    layer_outs = {name: func([test_input]) for name, func in functors.items()}
    return layer_outs


def get_layer_inputs(model, test_input, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_input)

    inputs = []

    for i in range(len(outs)):
        weights, biases = model.layers[i].get_weights()

        inputs_for_layer = []

        for input_index in range(len(test_input)):
            inputs_for_layer.append(np.add(np.dot(outs[i - 1][0][input_index] if i > 0 else test_input[input_index], weights), biases))

        inputs.append(inputs_for_layer)

    return [inputs[i] for i in range(len(inputs)) if i not in skip]


def get_python_version():
    if (sys.version_info > (3, 0)):
        # Python 3 code in this block
        return 3
    else:
        # Python 2 code in this block
        return 2


def show_image(vector):
    img = vector
    plt.imshow(img)
    plt.show()


def calculate_prediction_metrics(Y_test, Y_pred, score):
    """
    Calculate classification report and confusion matrix
    :param Y_test:
    :param Y_pred:
    :param score:
    :return:
    """
    # Find test and prediction classes
    Y_test_class = np.argmax(Y_test, axis=1)
    Y_pred_class = np.argmax(Y_pred, axis=1)

    classifications = np.absolute(Y_test_class - Y_pred_class)

    correct_classifications = []
    incorrect_classifications = []
    for i in range(1, len(classifications)):
        if (classifications[i] == 0):
            correct_classifications.append(i)
        else:
            incorrect_classifications.append(i)

    # Accuracy of the predicted values
    print(classification_report(Y_test_class, Y_pred_class))
    print(confusion_matrix(Y_test_class, Y_pred_class))

    acc = sum([np.argmax(Y_test[i]) == np.argmax(Y_pred[i]) for i in range(len(Y_test))]) / len(Y_test)
    v1 = ceil(acc * 10000) / 10000
    v2 = ceil(score[1] * 10000) / 10000
    correct_accuracy_calculation = v1 == v2
    try:
        if not correct_accuracy_calculation:
            raise Exception("Accuracy results don't match to score")
    except Exception as error:
        print("Caught this error: " + repr(error))


def get_dummy_dominants(model, dominants):
    import random
    # dominant = {x: random.sample(range(model.layers[x].output_shape[1]), 2) for x in range(1, len(model.layers))}
    dominant = {x: random.sample(range(0, 10), len(dominants[x])) for x in range(1, len(dominants) + 1)}
    return dominant

def save_quantization(qtized, filename):
    with h5py.File(filename + '_quantization.h5', 'w') as hf:
        hf.create_dataset("q", data=qtized)

    return

def load_quantization(filename):
    with h5py.File(filename + '_quantization.h5', 'r') as hf:
        qtized = hf["q"][:]

    return qtized


def save_max_comb(max_num, filename):
    with h5py.File(filename + '_max_comb.h5', 'w') as hf:
        hf.create_dataset("comb", data=[max_num])

    print("Max number of combinations saved to %s" %(filename))
    return

def load_max_comb(filename):
    with h5py.File(filename + '_max_comb.h5', 'r') as hf:
        max_num = hf["comb"][:]

    print("Max number of combinations loaded from %s" %(filename))
    return max_num


def save_data(data, filename):
    with h5py.File(filename + '_dataset.h5', 'w') as hf:
        hf.create_dataset("dataset", data=data)

    print("Data saved to %s" %(filename))
    return


def load_data(filename):
    with h5py.File(filename + '_dataset.h5', 'r') as hf:
        dataset = hf["dataset"][:]

    print("Data loaded from %s" %(filename))
    return dataset


def save_layerwise_relevances(relevant_neurons, filename):
    with h5py.File(filename + '_relevant_neurons.h5', 'w') as hf:
        hf.create_dataset("relevant_neurons",
                        data=relevant_neurons)

    return

def load_layerwise_relevances(filename):
    with h5py.File(filename + '_relevant_neurons.h5',
                    'r') as hf:
        relevant_neurons = hf["relevant_neurons"][:]

    print("Layerwise relevances loaded from %s" %(filename))

    return relevant_neurons


def save_data(data, filename):
    with h5py.File(filename + '_dataset.h5', 'w') as hf:
        hf.create_dataset("dataset", data=data)

    return


def load_data(filename):
    with h5py.File(filename + '_dataset.h5', 'r') as hf:
        dataset = hf["dataset"][:]

    return dataset


def save_perturbed_test(x_perturbed, y_perturbed, filename):
    # save X
    with h5py.File(filename + '_perturbations_x.h5', 'w') as hf:
        hf.create_dataset("x_perturbed", data=x_perturbed)

    # save Y
    with h5py.File(filename + '_perturbations_y.h5', 'w') as hf:
        hf.create_dataset("y_perturbed", data=y_perturbed)

    print("Layerwise relevances saved to  %s" %(filename))
    return


def load_perturbed_test(filename):
    # read X
    with h5py.File(filename + '_perturbations_x.h5', 'r') as hf:
        x_perturbed = hf["x_perturbed"][:]

    # read Y
    with h5py.File(filename + '_perturbations_y.h5', 'r') as hf:
        y_perturbed = hf["y_perturbed"][:]

    return x_perturbed, y_perturbed


def save_perturbed_test_groups(x_perturbed, y_perturbed, filename, group_index):
    # save X
    filename = filename + '_perturbations.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("x_perturbed", data=x_perturbed)
        group.create_dataset("y_perturbed", data=y_perturbed)

    print("Classifications saved in ", filename)

    return


def load_perturbed_test_groups(filename, group_index):
    with h5py.File(filename + '_perturbations.h5', 'r') as hf:
        group = hf.get('group' + str(group_index))
        x_perturbed = group.get('x_perturbed').value
        y_perturbed = group.get('y_perturbed').value

        return x_perturbed, y_perturbed


def create_experiment_dir(experiment_path, model_name,
                          selected_class, step_size,
                          approach, susp_num, repeat):
    # define experiments name, create directory experiments directory if it
    # doesnt exist
    experiment_name = model_name + '_C' + str(selected_class) + '_SS' + \
                      str(step_size) + '_' + approach + '_SN' + str(susp_num) + '_R' + str(repeat)

    if not path.exists(experiment_path):
        makedirs(experiment_path)

    return experiment_name


def save_classifications(correct_classifications, misclassifications, filename, group_index):
    filename = filename + '_classifications.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("correct_classifications", data=correct_classifications)
        group.create_dataset("misclassifications", data=misclassifications)

    print("Classifications saved in ", filename)
    return


def load_classifications(filename, group_index):
    filename = filename + '_classifications.h5'
    print
    filename
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            correct_classifications = group.get('correct_classifications').value
            misclassifications = group.get('misclassifications').value

            print("Classifications loaded from ", filename)
            return correct_classifications, misclassifications
    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)


def save_layer_outs(layer_outs, filename, group_index):
    filename = filename + '_layer_outs.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(layer_outs)):
            group.create_dataset("layer_outs_" + str(i), data=layer_outs[i])

    print("Layer outs saved in ", filename)
    return


def load_layer_outs(filename, group_index):
    filename = filename + '_layer_outs.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            layer_outs = []
            while True:
                layer_outs.append(group.get('layer_outs_' + str(i)).value)
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except (AttributeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("Layer outs loaded from ", filename)
        return layer_outs



def save_original_inputs(original_inputs, filename, group_index):
    filename = filename + '_originals.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("x_original", data=original_inputs)

    print("Originals saved in ", filename)

    return

def filter_correct_classifications(model, X, Y):
    X_corr = []
    Y_corr = []
    X_misc = []
    Y_misc = []
    for x, y in zip(X, Y):
        p = model.predict(np.expand_dims(x,axis=0))
        if np.argmax(p) == np.argmax(y):
            X_corr.append(x)
            Y_corr.append(y)
        else:
            X_misc.append(x)
            Y_misc.append(y)
    return np.array(X_corr), np.array(Y_corr), np.array(X_misc), np.array(Y_misc)


def filter_val_set(desired_class, X, Y):
    """
    Filter the given sets and return only those that match the desired_class value
    :param desired_class:
    :param X:
    :param Y:
    :return:
    """
    X_class = []
    Y_class = []
    for x, y in zip(X, Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)
    print("Validation set filtered for desired class: " + str(desired_class))
    return np.array(X_class), np.array(Y_class)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def get_trainable_layers(model):
    trainable_layers = []
    for idx, layer in enumerate(model.layers):
        try:
            if 'input' not in layer.name and 'softmax' not in layer.name and \
                    'pred' not in layer.name:
                weights = layer.get_weights()[0]
                trainable_layers.append(model.layers.index(layer))
        except:
            pass

    #trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return trainable_layers


def construct_spectrum_matrices(model, trainable_layers,
                                correct_classifications, misclassifications,
                                layer_outs):
    scores = []
    num_cf = []
    num_uf = []
    num_cs = []
    num_us = []
    for tl in trainable_layers:
        num_cf.append(np.zeros(model.layers[tl].output_shape[1]))  # covered (activated) and failed
        num_uf.append(np.zeros(model.layers[tl].output_shape[1]))  # uncovered (not activated) and failed
        num_cs.append(np.zeros(model.layers[tl].output_shape[1]))  # covered and succeeded
        num_us.append(np.zeros(model.layers[tl].output_shape[1]))  # uncovered and succeeded
        scores.append(np.zeros(model.layers[tl].output_shape[1]))

    for tl in trainable_layers:
        layer_idx = trainable_layers.index(tl)
        all_neuron_idx = range(model.layers[tl].output_shape[1])
        test_idx = 0
        for l in layer_outs[tl][0]:
            covered_idx = list(np.where(l > 0)[0])
            uncovered_idx = list(set(all_neuron_idx) - set(covered_idx))
            # uncovered_idx = list(np.where(l <= 0)[0])
            if test_idx in correct_classifications:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            elif test_idx in misclassifications:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1

    return scores, num_cf, num_uf, num_cs, num_us


def cone_of_influence_analysis(model, dominants):
    hidden_layers = [l for l in dominants.keys() if len(dominants[l]) > 0]
    target_layer = max(hidden_layers)

    scores = []
    for i in range(1, target_layer + 1):
        scores.append(np.zeros(model.layers[i].output_shape[1]))

    for i in range(2, target_layer + 1)[::-1]:
        for j in range(model.layers[i].output_shape[1]):
            for k in range(model.layers[i - 1].output_shape[1]):
                relevant_weights = model.layers[i].get_weights()[0][k]
                if (j in dominants[i] or scores[i - 1][j] > 0) and relevant_weights[j] > 0:
                    scores[i - 2][k] += 1
                elif (j in dominants[i] or scores[i - 1][j] > 0) and relevant_weights[j] < 0:
                    scores[i - 2][k] -= 1
                elif j not in dominants[i] and scores[i - 1][j] < 0 and relevant_weights[j] > 0:
                    scores[i - 2][k] -= 1
                elif j not in dominants[i] and scores[i - 1][j] < 0 and relevant_weights[j] < 0:
                    scores[i - 2][k] += 1
    print(scores)
    return scores


def weight_analysis(model):
    threshold_weight = 0.1
    deactivatables = []
    for i in range(2, target_layer + 1):
        for k in range(model.layers[i - 1].output_shape[1]):
            neuron_weights = model.layers[i].get_weights()[0][k]
            deactivate = True
            for j in range(len(neuron_weights)):
                if neuron_weights[j] > threshold_weight:
                    deactivate = False

            if deactivate:
                deactivatables.append((i, k))

    return deactivatables

def percent(part, whole):
    if part == 0:
        return 0
    return float(part) / whole * 100

def percent_str(part, whole):
    return "{0}%".format(float(part) / whole * 100)


def generate_adversarial(original_input, method, model,
                         target=None, target_class=None, sess=None, **kwargs):
    if not hasattr(generate_adversarial, "attack_types"):
        generate_adversarial.attack_types = {
            'fgsm': FastGradientMethod,
            'jsma': SaliencyMapMethod,
            'cw': CarliniWagnerL2,
            'bim': BasicIterativeMethod
        }

    if sess is None:
        sess = K.get_session()

    if method in generate_adversarial.attack_types:
        attacker = generate_adversarial.attack_types[method](KerasModelWrapper(model), sess)
    else:
        raise Exception("Method not supported")

    if type(original_input) is list:
        original_input = np.asarray(original_input)
    else:
        original_input = np.asarray([original_input])

        if target_class is not None:
            target_class = [target_class]

    if target is None and target_class is not None:
        target = np.zeros((len(target_class), model.output_shape[1]))
        target[np.arange(len(target_class)), target_class] = 1

    if target is not None:
        kwargs['y_target'] = target

    return attacker.generate_np(original_input, **kwargs)



