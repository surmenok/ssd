"""
Folder structure after preprocessing vis dataset.kitti.prepare.maybe_prepare_kitti_data:

kitti_2d
- raw
- video-split
-- 2011_09_26_0001
--- 000000004_004958.png (symlink)
--- ...
-- ...
- train
-- images
--- 000000.png (symlink)
--- ...
-- labels
--- 000000.txt (symlink)
--- ...
- val
-- images
--- 000012.png (symlink)
-- labels
--- 000012.txt (symlink)

Format of ground truth object detection files
http://www.cvlibs.net/datasets/kitti/eval_object.php
Per devkit docs:
All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""

import csv
import cv2
import os
from pathlib import Path
import torch
import torch.utils.data
from typing import Union


class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, path: Union[str, Path], data_transform=None):
        self.path = Path(path)
        self.image_ids = self._get_image_ids()

        self.labels = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.label_to_id = {l: i for i, l in enumerate(self.labels)}

        self.data_transform = data_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):
        if index < 0 or index > len(self.image_ids) - 1:
            raise ValueError('index is out of range.')
        image_id = self.image_ids[index]
        sample = self._get_example(image_id)
        if self.data_transform:
            sample = self.data_transform(sample)
        return sample

    def get_label_id(self, label):
        return self.label_to_id[label]

    def get_filename(self, idx):
        return self.path / 'images' / f'{self.image_ids[idx]}.png'

    def get_original_image(self, idx):
        return self._get_example(self.image_ids[idx])[0]

    def collate_fn(self, batch):
        # Need this as shapes of targets are different in different examples
        # Default collate function doesn't handle this
        imgs = [x[0] for x in batch]
        ys = [x[1] for x in batch]
        inputs = torch.stack(imgs)
        return inputs, ys

    def _get_image_ids(self):
        image_filenames = os.listdir(self.path / 'images')
        # TODO: Split extension by . instead of [:-4]
        image_ids = [filename[:-4] for filename in image_filenames if filename.endswith('.png')]
        return sorted(image_ids)

    def _get_example(self, image_id: int):
        image_path = self.path / 'images' / f'{image_id}.png'
        image = self._open_image(image_path)

        detections_filepath = self.path / 'labels' / f'{image_id}.txt'
        detections = self._get_detections(detections_filepath)
        y = self._transform_detections(detections)

        return image, y

    def _get_detections(self, detections_filepath: Path):
        detections = []

        with open(detections_filepath) as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                x1, y1, x2, y2 = map(float, row[4:8])
                label = row[0]

                if label == 'DontCare':
                    continue

                if x1 >= x2 or y1 >= y2:
                    raise Exception('Incorrect box: ({}, {}, {}, {})'.format(x1, y1, x2, y2))

                detections.append({
                    'label': label,
                    'left': x1,
                    'right': x2,
                    'top': y1,
                    'bottom': y2
                })

        return detections

    def _transform_detections(self, detections):
        boxes = []
        classes = []
        for d in detections:
            boxes.append([d['top'], d['left'], d['bottom'], d['right']])
            classes.append(self.get_label_id(d['label']))
        return torch.FloatTensor(boxes), torch.LongTensor(classes)

    def _open_image(self, filename: Path):
        """ Opens an image using OpenCV given the file path.

        Arguments:
            fn: the file path of the image

        Returns:
            The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
        """
        flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
        if not os.path.exists(filename):
            raise OSError('No such file or directory: {}'.format(filename))
        elif os.path.isdir(filename):
            raise OSError('Is a directory: {}'.format(filename))
        else:
            try:
                im = cv2.imread(str(filename), flags)
                if im is None:
                    raise OSError(f'File not recognized by opencv: {filename}')
                return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise OSError('Error handling image at: {}'.format(filename)) from e
