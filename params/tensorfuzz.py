from params.parameters import Parameters

tensorfuzz = Parameters()

tensorfuzz.tf_num_mutations = 64
tensorfuzz.tf_sigma = 0.2 * (255-0)/(1-(-1))
tensorfuzz.constraint = None