from src.deephunter import DeepHunter

def deephunter(params, experiment):
    dh = DeepHunter(params, experiment)
    dh.run()