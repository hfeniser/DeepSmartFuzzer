from pyflann import FLANN
import numpy as np

from coverages.coverage import AbstractCoverage
from coverages.utils import get_layer_outs_new


class TFCoverage(AbstractCoverage):

    def __init__(self, model, subject_layer, distance_threshold):
        self.model = model
        self.distant_vectors = []
        self.subject_layer = subject_layer
        self.distance_threshold = distance_threshold

    def get_measure_state(self):
        return self.distant_vectors

    def set_measure_state(self, vectors):
        self.distant_vectors = vectors

    def reset_measure_state(self):
        self.distant_vectors = []

    def get_current_coverage(self, with_implicit_reward=False):
        return len(self.distant_vectors)

    def test(self, test_inputs, with_implicit_reward=False):

        pen_layer_outs = get_layer_outs_new(self.model, test_inputs)[self.subject_layer]

        flann = FLANN()

        for plo in pen_layer_outs:
            if len(self.distant_vectors) > 0:
                _, dists = flann.nn(np.array(self.distant_vectors), plo, 1)
                if dists > self.distance_threshold:
                  self.distant_vectors.append(plo)
            else:
                self.distant_vectors.append(plo)

        return len(self.distant_vectors), self.distant_vectors
