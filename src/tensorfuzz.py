import random
import numpy as np

# TODO: Take below parameters from user
glob_num_iteration = 1000
glob_num_mutations = 100


class CorpusElement(object):
    """Class representing a single element of a corpus."""

    def __init__(self, data, parent):
        """Inits the object.

        Args:
          data: a list of numpy arrays representing the mutated data.
          parent: a reference to the CorpusElement this element is a mutation of.
        Returns:
          Initialized object.
        """
        self.data = data
        self.parent = parent

    def oldest_ancestor(self):
        """Returns the least recently created ancestor of this corpus item."""
        current_element = self
        generations = 0
        while current_element.parent is not None:
            current_element = current_element.parent
            generations += 1
        return current_element, generations


class Tensorfuzz:
    def __init__(self, corpus, coverage):
        self.num_iteration = glob_num_iteration
        self.corpus = corpus
        self.coverage = coverage
        self.last_coverage_state = None

    def fuzz(self):

        for iteration in range(self.num_iteration):
            if iteration % 100 == 0:
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


def do_basic_mutations(corpus_element, mutations_count, constraint=None, a_min=0, a_max=255):
    if len(corpus_element.data) > 1:
        inputs = corpus_element.data
        inp_batch = np.tile(inputs, [mutations_count, 1, 1, 1])
    else:
        inputs = corpus_element.data[0]
        inp_batch = np.tile(inputs, [mutations_count] + list(inputs.shape))

    sigma = 0.2
    noise = np.random.normal(size=inp_batch.shape, scale=sigma)

    if constraint is not None:
        # (image - original_image) is a single image. it gets broadcast into a batch
        # when added to 'noise'
        ancestor = corpus_element.oldest_ancestor()
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
