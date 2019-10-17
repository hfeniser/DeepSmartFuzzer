import numpy as np
import itertools
import src.image_transforms as image_transforms
from params.parameters import Parameters

deephunter = Parameters()

deephunter.K = 64
deephunter.batch1 = 64
deephunter.batch2 = 16
deephunter.p_min = 0.01
deephunter.gamma = 5
deephunter.alpha = 0.1
deephunter.beta = 0.5
deephunter.TRY_NUM = 100

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

deephunter.G = translation + rotation
deephunter.P = contrast + brightness + blur

deephunter.save_batch = False