from src.deephunter import DeepHunter

def deephunter(params, experiment):
    DeepHunter(experiment.dataset["test_inputs"], experiment.coverage)