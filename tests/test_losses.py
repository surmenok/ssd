import constants
import losses
import torch
from torch import tensor as T
import unittest


class TestSSDLoss(unittest.TestCase):
    def test_ssd_loss(self):
        grid_size = 4
        anchors = losses.create_anchors(grid_size)
        loss = losses.SSDLoss(anchors, 1. / grid_size, constants.TRANSFORMED_IMAGE_SIZE, num_classes=10)

        # TODO: pass activations and target through the loss and check results
        # Variations:
        # 1. Activations match target exactly, loss is 0
        # 2. Coordinates match but classes don't
        # 3. Classes match but coordinates don't

    def test_create_anchors(self):
        grid_size = 4
        anchors = losses.create_anchors(grid_size)
        self.assertEqual(anchors.shape, (16, 4))

        self.assertEqual(anchors[:, 2].numpy().min(), 0.25)
        self.assertEqual(anchors[:, 2].numpy().max(), 0.25)

        self.assertEqual(anchors[:, 3].numpy().min(), 0.25)
        self.assertEqual(anchors[:, 3].numpy().max(), 0.25)

        self.assertEqual(anchors[0, 0], 0.125)
        self.assertEqual(anchors[0, 1], 0.125)

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
