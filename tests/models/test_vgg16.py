from models.vgg import vgg16
import torch
import unittest


class TestUtils(unittest.TestCase):
    def test_forward_pass_vgg16(self):
        # This is not really a test. Just printing out chapes of tensors.
        # Input shape in SSD paper (300x300) is different from VGG paper (224x224)
        # Here we'll find dimensions of output of each layer of VGG-16 and compare with architecture in SSD paper
        # Turns out, SSD paper expects base model output shape 512x38x38 while VGG-16 from torchvision
        # outputs 512x37x37 on the same input size.
        # MaxPool2d layer reduces the output dimension if the input dimension is odd/
        batch_size = 4
        num_channels = 3
        image_size = (300, 300)

        base_model = vgg16(pretrained=True)

        input = torch.zeros((batch_size, num_channels, image_size[0], image_size[1]), dtype=torch.float32)

        x = input
        for f in list(base_model.features.modules())[1:]:
            x = f(x)
            print(f)
            print(x.shape)


if __name__ == '__main__':
    unittest.main()
