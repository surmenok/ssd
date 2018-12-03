import constants
import os
from pathlib import Path
import torch
import utils
import unittest


class TestUtils(unittest.TestCase):
    def test_load_dataset_success(self):
        dataset_path = Path(os.path.normpath(os.path.join(utils.__file__, '../../data/kitti_2d')))
        train_ds, val_ds = utils.load_dataset(dataset_path)

        self.assertEqual(len(train_ds), 6610)
        self.assertEqual(len(val_ds), 871)

        x, y = val_ds[0]

        # Check type and shape of x
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(x.shape, (3, constants.TRANSFORMED_IMAGE_SIZE[0], constants.TRANSFORMED_IMAGE_SIZE[1]))

        # Check that pixel values are normalized
        self.assertGreater(x.min(), -3)
        self.assertLess(x.max(), 3)

        self.assertEqual(len(y), 2)

        # Check type and shape of y[0] - boxes
        self.assertIsInstance(y[0], torch.Tensor)
        self.assertEqual(y[0].dtype, torch.float32)
        self.assertEqual(len(y[0].shape), 2)
        self.assertEqual(y[0].shape[1], 4)

        # Check type and shape of y[1] - classes
        self.assertIsInstance(y[1], torch.Tensor)
        self.assertEqual(y[1].dtype, torch.int64)
        self.assertEqual(len(y[1].shape), 1)

        # Check that number of boxes and classes are equal
        self.assertEqual(y[0].shape[0], y[1].shape[0])


if __name__ == '__main__':
    unittest.main()
