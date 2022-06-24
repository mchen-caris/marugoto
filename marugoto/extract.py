# %%
import os
import re
import json
from typing import Optional
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import PIL
from torchvision import models, transforms
from torch import nn
import numpy as np
from tqdm import tqdm
import h5py

__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'mvantreeck@ukaachen.de'


__changelog__ = {
    '0.2.0': 'Add extractor information to tiles.',
}

__all__ = ['extract_features_']


class SlideTileDataset(Dataset):
    def __init__(self, slide_dir: Path, transform=None, *, repetitions: int = 1) -> None:
        self.tiles = list(slide_dir.glob('*.jpg'))*repetitions
        assert self.tiles, f'no tiles found in {slide_dir}'
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.open(self.tiles[i])
        if self.transform:
            image = self.transform(image)

        return image


def _get_coords(filename) -> Optional[np.ndarray]:
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', str(filename)):
        coords = tuple(map(int, matches.groups()))
        assert len(coords) == 2, 'Error extracting coordinates'
        return np.array(coords)
    else:
        return None


def extract_features_(
    *slide_tile_paths: Path, outdir: Path, augmented_repetitions: int = 0,
    model=None, model_name=None,
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not model:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Identity()
        model_name = 'resnet18-imagenet'
    elif not model_name:
        raise ValueError('`model_name` has to specified when supplying a custom model.')

    model = model.eval().to(device)
    normal_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    augmenting_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomVerticalFlip(p=.5),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=.1, contrast=.2, saturation=.25, hue=.125)], p=.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    extractor_string = f'marugoto-extract-v{__version__}_{model_name}'
    with open(outdir/'info.json') as f:
        json.dump(f, {'extractor': extractor_string})

    for slide_tile_path in tqdm(slide_tile_paths):
        slide_tile_path = Path(slide_tile_path)
        # check if h5 for slide already exists / slide_tile_path path contains tiles
        if (h5outpath := outdir/f'{slide_tile_path.name}.h5').exists():
            print(f'{h5outpath} already exists.  Skipping...')
            continue
        if not next(slide_tile_path.glob('*.jpg'), False):
            print(f'No tiles in {slide_tile_path}.  Skipping...')
            continue

        ds = ConcatDataset([
            SlideTileDataset(slide_tile_path, normal_transform),
            SlideTileDataset(slide_tile_path, augmenting_transform,
                                repetitions=augmented_repetitions),
        ])
        dl = torch.utils.data.DataLoader(
            ds, batch_size=64, shuffle=False, num_workers=os.cpu_count(), drop_last=False)

        feats = []
        for batch in tqdm(dl, leave=False):
            feats.append(model(batch.half().to(device)).half().cpu().detach())

        with h5py.File(h5outpath, 'w') as f:
            f['coords'] = [_get_coords(fn) for fn in ds.tiles]
            f['feats'] = torch.concat(feats).cpu().numpy()
            f['augmented'] = np.repeat([False, True], [len(ds.datasets[0]), len(ds.datasets[1])])
            assert len(f['feats']) == len(f['augmented'])
            f.attrs['extractor'] = extractor_string


if __name__ == '__main__':
    import fire
    fire.Fire(extract_features_)

# %%
