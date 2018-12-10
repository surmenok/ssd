from models.resnet import resnet34
from models.ssd import SSDHead, SSDModel
import torch
import unittest


class TestUtils(unittest.TestCase):
    def test_forward_pass_ssd(self):
        num_labels = 10
        k = 1
        batch_size = 4
        num_channels = 3
        image_size = (224, 224)

        base_model = resnet34(pretrained=True)
        head = SSDHead(k, num_labels, -3.)
        ssd = SSDModel(base_model, head)

        input = torch.zeros((batch_size, num_channels, image_size[0], image_size[1]), dtype=torch.float32)

        loc_flattened, class_flattened = ssd(input)


if __name__ == '__main__':
    unittest.main()
