import constants
import losses
from models.resnet import resnet34
from models.ssd import SSDHead, SSDModel
import torch
from torch import tensor as T
import unittest


class TestSSDLoss(unittest.TestCase):
    def test_ssd_loss(self):
        anchors = losses.create_anchors(grid_sizes=[(7, 7), (4, 4), (2, 2), (1, 1)], zoom_levels=[1], aspect_ratios=[1])
        self.assertEqual(anchors.size(), (70, 4))

        loss = losses.SSDLoss(anchors, constants.TRANSFORMED_IMAGE_SIZE, num_classes=10)

        num_labels = 10
        k = 1
        batch_size = 4
        num_channels = 3
        image_size = (224, 224)

        base_model = resnet34(pretrained=True)
        head = SSDHead(k, num_labels, -3.)
        ssd = SSDModel(base_model, head)

        input = torch.zeros((batch_size, num_channels, image_size[0], image_size[1]), dtype=torch.float32)
        out = ssd(input)
        self.assertEqual(out[0].size(), (4, 70, 4))
        self.assertEqual(out[1].size(), (4, 70, 11))

        target = [(T([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32), T([1, 2], dtype=torch.long)),
                  (T([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32), T([1, 2], dtype=torch.long)),
                  (T([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32), T([1, 2], dtype=torch.long)),
                  (T([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32), T([1, 2], dtype=torch.long))]

        l = loss.loss(out, target)

        # Sanity check: nothing fails, returns dict of expected structure
        self.assertEqual(len(l), 3)
        self.assertTrue('classification' in l)
        self.assertTrue('localization' in l)
        self.assertTrue('total' in l)

    def test_create_anchors_simple(self):
        grid_size = (4, 4)
        anchors = losses.create_anchors([grid_size], [1], [1])
        self.assertEqual(anchors.shape, (16, 4))

        # Check height
        self.assertEqual(anchors[:, 2].numpy().min(), 0.25)
        self.assertEqual(anchors[:, 2].numpy().max(), 0.25)

        # Check width
        self.assertEqual(anchors[:, 3].numpy().min(), 0.25)
        self.assertEqual(anchors[:, 3].numpy().max(), 0.25)

        # Check center
        self.assertEqual(anchors[0, 0], 0.125)
        self.assertEqual(anchors[0, 1], 0.125)

    def test_create_anchors_1_2(self):
        grid_size = (2, 2)
        anchors = losses.create_anchors_for_one_level(grid_size, [0.5, 1, 2], [0.5, 1, 2])
        print(anchors)
        self.assertEqual(anchors.shape, (36, 4))

        # TODO: add more checks of anchors

    def test_box_hw_to_corners(self):
        box_hw = T([[1., 2., 0.5, 0.7],
                    [3.5, 4., 1.5, 2.]])
        corners = losses.box_hw_to_corners(box_hw)

        expected_corners = T([[0.75, 1.65, 1.25, 2.35],
                              [2.75, 3., 4.25, 5.]])

        error = float(((expected_corners - corners) * (expected_corners - corners)).sum())
        self.assertEqual(error, 0)

    def test_activation_to_bbox_corners(self):
        pass


if __name__ == '__main__':
    unittest.main()
