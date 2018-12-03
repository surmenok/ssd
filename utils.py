from pathlib import Path
from typing import Tuple
from dataset.kitti import KITTIDataset
from typing import Union
from val_transforms import get_transforms


def load_dataset(path: Union[str, Path]) -> Tuple[KITTIDataset, KITTIDataset]:
    path = Path(path)
    data_transform = get_transforms()
    train_ds = KITTIDataset(path / 'train', data_transform)
    val_ds = KITTIDataset(path / 'val', data_transform)
    return train_ds, val_ds
