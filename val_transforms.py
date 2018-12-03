import constants
from torch import tensor as T
import torchvision
from transforms import Normalize, Resize, ToTensor, to_numpy_image


def get_transforms():
    data_transform = torchvision.transforms.Compose([
        ToTensor(),
        Normalize(mean=constants.DATA_MEAN, std=constants.DATA_STD),
        Resize(constants.TRANSFORMED_IMAGE_SIZE)
    ])
    return data_transform


def detransform(transformed_tensor, device='cpu'):
    # This method doesn't modify input data
    # Slower than in-place normalization of torchvision.transforms.Normalize
    mean_tensor = T(constants.DATA_MEAN).reshape(3, 1, 1).to(device)
    std_tensor = T(constants.DATA_STD).reshape((3, 1, 1)).to(device)
    detransformed_tensor = transformed_tensor.mul(std_tensor).add(mean_tensor)
    return to_numpy_image(detransformed_tensor)
