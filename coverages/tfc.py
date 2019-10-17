from pyflann import FLANN
import numpy as np

from coverages.coverage import AbstractCoverage
from coverages.utils import get_layer_outs_new

_BUFFER_SIZE = 100

class TFCoverage(AbstractCoverage):

    def __init__(self, model, subject_layer, distance_threshold):
        self.model = model
        self.distant_vectors = []
        self.distant_vectors_buffer = []
        self.subject_layer = subject_layer
        self.distance_threshold = distance_threshold
        self.flann = FLANN()

    def get_measure_state(self):
        s = []
        s.append(self.distant_vectors)
        s.append(self.distant_vectors_buffer)
        return s

    def set_measure_state(self, s):
        self.distant_vectors = s[0]
        self.distant_vectors_buffer = s[1]
        if len(self.distant_vectors_buffer) > _BUFFER_SIZE:
            self.build_index_and_flush_buffer()

    def reset_measure_state(self):
        self.flann.delete_index()
        self.distant_vectors = []
        self.distant_vectors_buffer = []

    def get_current_coverage(self, with_implicit_reward=False):
        return len(self.distant_vectors)

    def build_index_and_flush_buffer(self):
        self.distant_vectors_buffer = []
        self.flann.build_index(np.array(self.distant_vectors))


    def test(self, test_inputs, with_implicit_reward=False):
        pen_layer_outs = get_layer_outs_new(self.model, test_inputs)[self.subject_layer]

        for plo in pen_layer_outs:
            if len(self.distant_vectors) > 0:
                _, approx_distances = self.flann.nn_index(plo, 1)
                exact_distances = [
                    np.sum(np.square(plo - distant_vec)) 
                    for distant_vec in self.distant_vectors_buffer
                    ]
                nearest_distance = min(exact_distances + approx_distances.tolist())
                if nearest_distance > self.distance_threshold:
                    self.distant_vectors_buffer.append(plo)
                    self.distant_vectors.append(plo)
            else:
                self.flann.build_index(plo)
                self.distant_vectors.append(plo)

        return len(self.distant_vectors), self.distant_vectors
