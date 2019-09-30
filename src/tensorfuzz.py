from src.deephunter import f
import time
import pyflann
import random
import numpy as np

# TODO: Take below parameters from user
glob_num_iteration = 1000
glob_num_mutations = 100


class Tensorfuzz:
    def __init__(self, corpus, coverage):
        self.num_iteration = glob_num_iteration
        self.corpus = corpus
        self.coverage = coverage
        self.last_coverage_state = None

    def fuzz(self):
        for iteration in range(self.num_iteration):
            if iteration % 100 == 0:
                print("FUZZING YEY!")
                print("fuzzing iteration: %s", iteration)

            inp = uniform_sample_function(self.corpus)
            # inp = recent_sample_function(self.corpus)
            mutated_data_batches = do_basic_mutations(inp, glob_num_mutations)

            prospective_corpus = self.corpus.copy()
            np.append(prospective_corpus, mutated_data_batches)

            last_cov, cov = self.coverage.step(prospective_corpus, update_state=False,
                                               coverage_state=self.last_coverage_state)

            if cov > 0:
                print("Coverage gain: ", cov)
                print()
                self.corpus = prospective_corpus.copy()
                self.last_coverage_state = last_cov

        return None


def uniform_sample_function(corpus):
    choice = random.choice(corpus)
    return choice


def recent_sample_function(corpus):
    reservoir = corpus[-5:] + [random.choice(corpus)]
    choice = random.choice(reservoir)
    return choice


def do_basic_mutations(inp, mutations_count, constraint=None, a_min=-1.0, a_max=1.0):
    inp_batch = np.tile(inp, [mutations_count] + list(inp.shape))
    sigma = 0.2
    noise = np.random.normal(size=inp_batch.shape, scale=sigma)

    if constraint is not None:
        # (image - original_image) is a single image. it gets broadcast into a batch
        # when added to 'noise'
        ancestor, _ = inp.oldest_ancestor()
        original_image = ancestor.data[0]
        original_image_batch = np.tile(
            original_image, [mutations_count, 1, 1, 1]
        )
        cumulative_noise = noise + (inp_batch - original_image_batch)
        noise = np.clip(cumulative_noise, a_min=-constraint, a_max=constraint)
        mutated_image_batch = noise + original_image_batch
    else:
        mutated_image_batch = noise + inp_batch

    mutated_image_batch = np.clip(
        mutated_image_batch, a_min=a_min, a_max=a_max
    )

    mutated_batches = [mutated_image_batch]

    return mutated_batches
