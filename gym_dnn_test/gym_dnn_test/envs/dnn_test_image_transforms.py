from gym_dnn_test.envs.image_transforms import *
from gym_dnn_test.envs.dnn_test_base import DNN_Test_Base

class DNN_Test_Image_Transforms(DNN_Test_Base):
    def apply_action(self, action):
        print("self.mutated_input.shape", self.mutated_input.shape)
        self.mutated_input = self.mutated_input.reshape(28,28,1)
        if action == 0:
            self.mutated_input = image_translation(self.mutated_input, (10,10))
        elif action == 1:
            self.mutated_input = image_translation(self.mutated_input, (30,30))
        elif action == 2:
            self.mutated_input = image_translation(self.mutated_input, (100,100))
        elif action == 3:
            self.mutated_input = image_scale(self.mutated_input, (1.5,1.5))
        elif action == 4:
            self.mutated_input = image_scale(self.mutated_input, (3.5,3.5))
        elif action == 5:
            self.mutated_input = image_scale(self.mutated_input, (6.0,6.0))
        elif action == 6:
            self.mutated_input = image_shear(self.mutated_input, (-1.0,0))
        elif action == 7:
            self.mutated_input = image_shear(self.mutated_input, (-0.3,0))
        elif action == 8:
            self.mutated_input = image_shear(self.mutated_input, (-0.1,0))
        elif action == 9:
            self.mutated_input = image_rotation(self.mutated_input, 3)
        elif action == 10:
            self.mutated_input = image_rotation(self.mutated_input, 10)
        elif action == 11:
            self.mutated_input = image_rotation(self.mutated_input, 30)
        elif action == 12:
            self.mutated_input = image_contrast(self.mutated_input, 1.2)
        elif action == 13:
            self.mutated_input = image_contrast(self.mutated_input, 2.1)
        elif action == 14:
            self.mutated_input = image_contrast(self.mutated_input, 3.0)
        elif action == 15:
            self.mutated_input = image_brightness(self.mutated_input, 10)
        elif action == 16:
            self.mutated_input = image_brightness(self.mutated_input, 30)
        elif action == 17:
            self.mutated_input = image_brightness(self.mutated_input, 100)
        elif action == 18:
            self.mutated_input = image_blur(self.mutated_input, 1)
        elif action == 19:
            self.mutated_input = image_blur(self.mutated_input, 4)
        elif action == 20:
            self.mutated_input = image_blur(self.mutated_input, 7)
        else:
            raise Exception('Unknown action:'+str(action))
        self.mutated_input = self.mutated_input.reshape(-1,28,28,1)
        print("self.mutated_input.shape2", self.mutated_input.shape)