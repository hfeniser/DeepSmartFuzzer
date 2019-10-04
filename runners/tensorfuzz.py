from src.tensorfuzz import Tensorfuzz, CorpusElement


def tensorfuzz(params, experiment):
    tf = Tensorfuzz(seed_corpus_from_numpy_arrays(experiment.dataset["test_inputs"]), experiment.coverage)
    tf.fuzz()


def seed_corpus_from_numpy_arrays(numpy_arrays):
    """Constructs a seed_corpus given numpy_arrays.

    We only use the first element of the batch that we fetch, because
    we're only trying to create one corpus element, and we may end up
    getting back a whole batch of coverage due to the need to tile our
    inputs to fit the static shape of certain feed_dicts.
    Args:
      numpy_arrays: multiple lists of input_arrays, each list with as many
        arrays as there are input tensors.
    Returns:
      List of CorpusElements.
    """
    seed_corpus = []
    for input_array_list in numpy_arrays:
        new_element = CorpusElement(input_array_list, None)
        seed_corpus.append(new_element)

    return seed_corpus
