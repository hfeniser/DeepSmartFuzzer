import numpy as np
import itertools
import src.image_transforms as image_transforms
from params.parameters import Parameters

deephunter_mnist = Parameters()

deephunter_mnist.input_shape = (1, 28, 28, 1)
deephunter_mnist.input_lower_limit = 0
deephunter_mnist.input_upper_limit = 255

deephunter_mnist.K = 64
deephunter_mnist.batch1 = 64
deephunter_mnist.batch2 = 16
deephunter_mnist.p_min = 0.01
deephunter_mnist.gamma = 5
deephunter_mnist.alpha = 0.1
deephunter_mnist.beta = 0.5
deephunter_mnist.TRY_NUM = 100

# translation = list(itertools.product([getattr(image_transforms,"image_translation")], [(10+10*k,10+10*k) for k in range(10)]))
# scale = list(itertools.product([getattr(image_transforms, "image_scale")], [(1.5+0.5*k,1.5+0.5*k) for k in range(10)]))
# shear = list(itertools.product([getattr(image_transforms, "image_shear")], [(-1.0+0.1*k,0) for k in range(10)]))
# rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], [3+3*k for k in range(10)]))
# contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [1.2+0.2*k for k in range(10)]))
# brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10+10*k for k in range(10)]))
# blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k+1 for k in range(10)]))

translation = list(itertools.product([getattr(image_transforms, "image_translation")],
                                    [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]))
rotation = list(
    itertools.product([getattr(image_transforms, "image_rotation")], [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15]))
contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [1.2 + 0.2 * k for k in range(10)]))
brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(10)]))
blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(10)]))

deephunter_mnist.G = translation + rotation
deephunter_mnist.P = contrast + brightness + blur

deephunter_mnist.save_batch = False