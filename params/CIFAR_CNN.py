from params.parameters import Parameters

CIFAR_CNN = Parameters()
CIFAR_CNN.tfc_threshold = 9

CIFAR_CNN.model_input_scale = [0,1]
CIFAR_CNN.skip_layers = [0,2,5,6,8,11,12,13,15]