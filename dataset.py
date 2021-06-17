import os
import numpy as np
import pandas as pd
import tifffile
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, augmentation, directory: str, df_rle: pd.DataFrame, df_imgs: pd.DataFrame):
        super().__init__()
        self.transform = transform
        self.augmentation = augmentation
        self.df_rle = df_rle
        self.df_imgs = df_imgs
        self.roots = [os.path.join(directory, image) for image in df_imgs['image_file'].tolist()]

    def rle_decode(df, image: np.ndarray, root: str):
        """ transform rle code to a segmentational mask"""

        if '/' in root:
            # root = 'directory/file.extension'
            image_name = root.split('/')[1].split('.')[0]
        else:
            # root = 'file.extension'
            image_name = root.split('.')[0]

        column = df.loc[df['id'] == image_name, 'encoding']

        code = list(map(int, column.values[0].split()))

        starts, length = [np.asarray(x) for x in (code[0::2], code[1::2])]
        starts = starts - 1
        ends = starts + length
        pixels = sum([list(range(s, e)) for s, e in zip(starts, ends)], [])

        shape = image.shape
        mask = np.ones(shape[0] * shape[1], dtype=int)
        mask[pixels] = 255

        mask = mask.reshape((shape[0], shape[1]), order='F')
        return image, mask

    def __len__(self):
        return len(self.images)

    def __iter__(self, x):
        return iter(x)

    def __getitem__(self, item):
        image_root = self.roots[item]
        image = tifffile.imread(image_root)
        image, mask = self.rle_decode(self.df_rle, image, image_root)
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


def encode_rle(mask) -> str:
    """convert mask to run-length encoded string"""

    h, w = mask.shape[0], mask.shape[1]
    mask = mask.reshape(h * w)
    code = []
    count = 0
    for i in range(len(mask)):
        if mask[i] == 255:
            count += 1
        else:
            code.extend([i, count])
    code = list(map(str, code))
    rle_str = ' '.join(code)
    return rle_str
