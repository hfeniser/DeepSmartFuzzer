import random
import numpy as np
from src.utility import init_image_plots, update_image_plots

glob_num_mutations = 64


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
    def __init__(self, params, experiment):
        self.params = params
        self.experiment = experiment
        self.corpus = self.experiment.corpus
        self.coverage = self.experiment.coverage
        self.input_shape = self.params.input_shape

        if self.params.image_verbose:
            self.f_current = init_image_plots(1, 1, self.input_shape)


    def fuzz(self):
        self.experiment.iteration = 0
        while not self.experiment.termination_condition():
            if self.params.verbose:
                print("fuzzing iteration: %s" % self.experiment.iteration)

            # inp = self.uniform_sample_function(self.corpus)
            inp = self.recent_sample_function(self.corpus)
            mutated_data_batches = self.do_basic_mutations(inp, self.params.tf_num_mutations)

            inputs_to_add = []
            for mdata in mutated_data_batches[0]:
                m_input = np.array(mdata).reshape(*self.input_shape)
                last_cov, coverage_gain = self.coverage.step(m_input, update_state=False)

                if coverage_gain > 0:
                    self.coverage.step(m_input, update_state=True, coverage_state=last_cov)
                    inputs_to_add.append(mdata)

                    if self.params.verbose:
                        print("coverage increase: %0.2f" % coverage_gain)

                if self.params.image_verbose:
                    title = "coverage increase: %0.2f" % coverage_gain
                    update_image_plots(self.f_current, m_input, title)
            
            for inpa in inputs_to_add:
                corpus_element = CorpusElement(inpa, inp)
                self.corpus.append(corpus_element)
            
            if len(inputs_to_add) > 0:
                self.experiment.input_chooser.append(np.array(inputs_to_add), np.array([-1]*len(inputs_to_add)))
            
            self.experiment.iteration += 1

        return None


    def uniform_sample_function(self, corpus):
        choice = random.choice(corpus)
        return choice


    def recent_sample_function(self, corpus):
        reservoir = corpus[-5:] + [random.choice(corpus)]
        choice = random.choice(reservoir)
        return choice


    def do_basic_mutations(self, corpus_element, mutations_count):
        if len(corpus_element.data) > 1:
            inputs = corpus_element.data
            inp_batch = np.tile(inputs, [mutations_count, 1, 1, 1])
        else:
            inputs = corpus_element.data[0]
            inp_batch = np.tile(inputs, [mutations_count] + list(inputs.shape))

        print('Input batch shape:')
        print(inp_batch.shape)
        noise = np.random.normal(size=inp_batch.shape, scale=self.params.tf_sigma)

        if self.params.constraint is not None:
            # (image - original_image) is a single image. it gets broadcast into a batch
            # when added to 'noise'
            ancestor = corpus_element.oldest_ancestor()
            original_image = ancestor.data[0]
            original_image_batch = np.tile(
                original_image, [mutations_count, 1, 1, 1]
            )
            cumulative_noise = noise + (inp_batch - original_image_batch)
            noise = np.clip(cumulative_noise, a_min=-self.params.constraint, a_max=self.params.constraint)
            mutated_image_batch = noise + original_image_batch
        else:
            mutated_image_batch = noise + inp_batch

        mutated_image_batch = np.clip(mutated_image_batch, a_min=self.params.input_lower_limit, a_max=self.params.input_upper_limit)

        print('Mutated img batch shape:')
        print(mutated_image_batch.shape)
        mutated_batches = [mutated_image_batch]

        return mutated_batches
