import numpy as np
import random

glob_num_iteration = 100
glob_num_mutations = 100

class Tensorfuzz():
    def __init__(self, mutation_function, corpus, coverage):
        self.num_iteration = glob_num_iteration
        self.corpus = corpus
        self.coverage = coverage

    def fuzz():
        for iteration in range(self.num_iteration):
            if iteration % 100 == 0:
                tf.logging.info("fuzzing iteration: %s", iteration)

            inp = uniform_sample_function(self.corpus)
            mutated_data_batches = do_basic_mutations(inp, glob_num_mutations)

            last_cov, cov = self.coverage.step(
                self.corpus.append(mutated_data_batches), update_stae=False)


            if cov > 0:
                self.corpus.append(mutated_data_batches)


        return None


def uniform_sample_function(corpus):
    choice = random.choice(corpus)
    return choice


def recent_sample_function(corpus):
    reservoir = corpus[-5:] + [random.choice(corpus)]
    choice = random.choice(reservoir)
    return choice


def do_basic_mutations(
    inp, mutations_count, constraint=None, a_min=-1.0, a_max=1.0
):
    image_batch = np.tile(inp, [mutations_count] + list(image.shape))

    sigma = 0.2
    noise = np.random.normal(size=image_batch.shape, scale=sigma)

    if constraint is not None:
        # (image - original_image) is a single image. it gets broadcast into a batch
        # when added to 'noise'
        ancestor, _ = corpus_element.oldest_ancestor()
        original_image = ancestor.data[0]
        original_image_batch = np.tile(
            original_image, [mutations_count, 1, 1, 1]
        )
        cumulative_noise = noise + (image_batch - original_image_batch)
        # pylint: disable=invalid-unary-operand-type
        noise = np.clip(cumulative_noise, a_min=-constraint, a_max=constraint)
        mutated_image_batch = noise + original_image_batch
    else:
        mutated_image_batch = noise + image_batch

    mutated_image_batch = np.clip(
        mutated_image_batch, a_min=a_min, a_max=a_max
    )

    mutated_batches = [mutated_image_batch]

    return mutated_batches

