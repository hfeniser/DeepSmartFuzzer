from src.tensorfuzz import Tensorfuzz

def tensorfuzz(params, experiment):
    tf = Tensorfuzz(experiment.dataset["test_inputs"], experiment.coverage)
    tf.fuzz()
