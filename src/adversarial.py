import numpy as np

def check_adversarial(experiment, params):
    print("checking adversarial")
    i = experiment.input_chooser.initial_nb_inputs
    if params.input_chooser == "clustered_random":
        new_inputs = experiment.input_chooser.test_inputs[i:]
        new_outputs = experiment.input_chooser.test_outputs[i:]
    else:
        new_inputs = experiment.input_chooser.features[i:]
        new_outputs = experiment.input_chooser.labels[i:]

    model = experiment.model

    predictions = model.predict(new_inputs)

    nb_adversarial = np.sum(np.argmax(predictions, axis=1) != new_outputs)
    nb_total = len(new_outputs)

    print("nb_adversarial: %g" % nb_adversarial)
    print("percent_adversarial: %g" % (nb_adversarial/nb_total))