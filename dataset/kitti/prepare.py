import numpy as np
import os
import re
import shutil
import zipfile


def extract_data(input_dir, output_dir):
    """
    Extract zipfiles at input_dir into output_dir

    Code from https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/prepare_kitti_data.py
    """
    if os.path.isdir(output_dir):
        print('Using extracted data at %s.' % output_dir)
        return

    for filename in (
            'data_object_label_2.zip',
            'data_object_image_2.zip',
            'devkit_object.zip'):
        filename = os.path.join(input_dir, filename)
        zf = zipfile.ZipFile(filename, 'r')
        print('Unzipping %s ...' % filename)
        zf.extractall(output_dir)


def get_image_to_video_mapping(devkit_dir):
    """
    Return a mapping from image filename (e.g. 7282 which is training/image_2/007282.png)
        to video and frame (e.g. {'video': '2011_09_26_0005', 'frame': 109})

    Code from https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/prepare_kitti_data.py
    """
    image_to_video = {}
    with open(os.path.join(devkit_dir, 'mapping', 'train_mapping.txt'), 'r') as infile:
        mapping_lines = infile.readlines()

    with open(os.path.join(devkit_dir, 'mapping', 'train_rand.txt'), 'r') as infile:
        for image_index, mapping_index in enumerate(infile.read().split(',')):
            mapping_index = mapping_index.strip()
            if not mapping_index:
                continue
            mapping_index = int(mapping_index) - 1
            map_line = mapping_lines[mapping_index]
            match = re.match('^\s*[\d_]+\s+(\d{4}_\d{2}_\d{2})_drive_(\d{4})_sync\s+(\d+)$\s*$', map_line)
            if not match:
                raise ValueError('Unrecognized mapping line "%s"' % map_line)
            date = match.group(1)
            video_id = match.group(2)
            video_name = '%s_%s' % (date, video_id)
            frame_index = int(match.group(3))
            if image_index in image_to_video:
                raise ValueError('Conflicting mappings for image %s' % image_index)
            image_to_video[image_index] = {
                'video': video_name,
                'frame': frame_index,
            }

    return image_to_video


def split_by_video(training_dir, mapping, split_dir,
                   use_symlinks=True):
    """
    Create one directory per video in split_dir

    Code from https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/prepare_kitti_data.py
    """
    new_images_dir = os.path.join(split_dir, 'images')
    new_labels_dir = os.path.join(split_dir, 'labels')
    if os.path.isdir(new_images_dir):
        shutil.rmtree(new_images_dir)
    if os.path.isdir(new_labels_dir):
        shutil.rmtree(new_labels_dir)

    for old_image_fname in os.listdir(os.path.join(training_dir, 'image_2')):
        old_image_path = os.path.abspath(os.path.join(training_dir, 'image_2', old_image_fname))
        image_index_str, image_ext = os.path.splitext(
            os.path.basename(old_image_fname))
        image_index_int = int(image_index_str)
        video_name = mapping[image_index_int]['video']
        frame_id = '%09d' % mapping[image_index_int]['frame']

        # Copy image
        new_image_dir = os.path.join(new_images_dir, video_name)
        if not os.path.isdir(new_image_dir):
            os.makedirs(new_image_dir)
        new_image_fname = '%s_%s%s' % (frame_id, image_index_str, image_ext)
        new_image_path = os.path.join(new_image_dir, new_image_fname)
        if use_symlinks:
            os.symlink(old_image_path, new_image_path)
        else:
            shutil.copyfile(old_image_path, new_image_path)

        # Copy label
        old_label_fname = '%s.txt' % image_index_str
        old_label_path = os.path.abspath(os.path.join(training_dir, 'label_2', old_label_fname))
        new_label_fname = '%s_%s.txt' % (frame_id, image_index_str)
        new_label_dir = os.path.join(new_labels_dir, video_name)
        if not os.path.isdir(new_label_dir):
            os.makedirs(new_label_dir)
        new_label_path = os.path.join(new_label_dir, new_label_fname)
        if use_symlinks:
            os.symlink(old_label_path, new_label_path)
        else:
            shutil.copyfile(old_label_path, new_label_path)


def split_for_training(split_dir, train_dir, val_dir,
                       train_split=0.9,
                       use_symlinks=True):
    """
    Create directories of images for training and validation

    Based on https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/prepare_kitti_data.py
    Modifications: random shuffle of videos instead of splitting by date.
    """
    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
    if os.path.isdir(val_dir):
        shutil.rmtree(val_dir)

    image_dirs = os.listdir(os.path.join(split_dir, 'images'))

    train_count = int(len(image_dirs) * train_split)
    prev_state = np.random.get_state()
    np.random.seed(42)
    np.random.shuffle(image_dirs)
    np.random.set_state(prev_state)
    val_dirs = image_dirs[train_count:]

    for images_dirname in os.listdir(os.path.join(split_dir, 'images')):
        if images_dirname in val_dirs:
            output_dir = val_dir
        else:
            output_dir = train_dir

        # Copy images
        old_images_dir = os.path.join(split_dir, 'images', images_dirname)
        new_images_dir = os.path.join(output_dir, 'images')
        if not os.path.isdir(new_images_dir):
            os.makedirs(new_images_dir)
        for fname in os.listdir(old_images_dir):
            old_image_path = os.path.realpath(os.path.join(old_images_dir, fname))
            new_image_path = os.path.join(new_images_dir, os.path.basename(old_image_path))
            if use_symlinks:
                os.symlink(old_image_path, new_image_path)
            else:
                shutil.move(old_image_path, new_image_path)

        # Copy labels
        old_labels_dir = os.path.join(split_dir, 'labels', images_dirname)
        new_labels_dir = os.path.join(output_dir, 'labels')
        if not os.path.isdir(new_labels_dir):
            os.makedirs(new_labels_dir)
        for fname in os.listdir(old_labels_dir):
            old_label_path = os.path.realpath(os.path.join(old_labels_dir, fname))
            new_label_path = os.path.join(new_labels_dir, os.path.basename(old_label_path))
            if use_symlinks:
                os.symlink(old_label_path, new_label_path)
            else:
                shutil.move(old_label_path, new_label_path)


def maybe_prepare_kitti_data(source_dir, output_dir, use_symlinks=True):
    """
    Code from https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/prepare_kitti_data.py
    """
    if os.path.exists(os.path.join(output_dir, 'train')):
        print('Training data already exists')
        return

    print('Extracting zipfiles ...')
    extract_data(
        source_dir,
        os.path.join(output_dir, 'raw'),
    )

    print('Calculating image to video mapping ...')
    mapping = get_image_to_video_mapping(
        os.path.join(output_dir, 'raw'),
    )

    print('Splitting images by video ...')
    split_by_video(
        os.path.join(output_dir, 'raw', 'training'),
        mapping,
        os.path.join(output_dir, 'video-split'),
        use_symlinks=use_symlinks,
    )

    print('Creating train/val split ...')
    split_for_training(
        os.path.join(output_dir, 'video-split'),
        os.path.join(output_dir, 'train'),
        os.path.join(output_dir, 'val'),
        use_symlinks=use_symlinks,
    )

    print('Done.')
