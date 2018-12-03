import constants
import torch
import losses
import unittest


class TestSSDLoss(unittest.TestCase):
    def test_ssd_loss(self):
        grid_size = 4
        anchors = losses.create_anchors(grid_size)
        loss = losses.SSDLoss(anchors, 1. / grid_size, constants.TRANSFORMED_IMAGE_SIZE, 10)

        # TODO: pass activations and target through the loss and check results
        # Variations:
        # 1. Activations match target exactly, loss is 0
        # 2. Coordinates match but classes don't
        # 3. Classes match but coordinates don't

    def test_create_anchors(self):
        pass

    def test_box_hw_to_corners(self):
        pass

    def test_activation_to_bbox_corners(self):
        pass


if __name__ == '__main__':
    unittest.main()
