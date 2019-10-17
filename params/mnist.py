
from params.parameters import Parameters

mnist = Parameters()

mnist.input_shape = (1, 28, 28, 1)
mnist.input_lower_limit = 0
mnist.input_upper_limit = 255