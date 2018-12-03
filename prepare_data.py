from dataset.kitti import maybe_prepare_kitti_data

if __name__ == '__main__':
    source_dir = '../data'
    output_dir = '../data/kitti_2d'
    maybe_prepare_kitti_data(source_dir, output_dir)
