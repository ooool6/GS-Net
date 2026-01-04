import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import logging

logger = logging.getLogger(__name__)

ORIGINAL_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
NUM_CLASSES = len(ORIGINAL_LABELS)


class SpineParseNetDataset(Dataset):
    def __init__(self, h5_file_path, mode='train', transform=None, use_unary=False, return_weight=False, num_classes=17, skip_mask=False):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.mode = mode.lower()
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got {mode}")
        self.transform = transform
        self.use_unary = use_unary
        self.return_weight = return_weight
        self.num_classes = num_classes
        self.skip_mask = skip_mask

        self.mri_data = self.h5_file[f'{self.mode}/mr']
        self.mask_data = self.h5_file[f'{self.mode}/mask'] if not skip_mask else None
        self.unary_data = self.h5_file[f'{self.mode}/unary'] if use_unary else None
        self.weight_data = self.h5_file[f'{self.mode}/weight'] if return_weight else None

        logger.info(f"Dataset in {self.mode} mode: Loading all {self.mri_data.shape[0]} slices.")
        logger.info(f"MRI data for {self.mode}: shape={self.mri_data.shape}, range=[{self.mri_data[0].min():.4f}, {self.mri_data[0].max():.4f}]")
        ORIGINAL_LABELS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16])
        if self.mask_data is not None:
            unique_values = np.unique(self.mask_data[0])
            logger.info(f"Mask data for {self.mode}: shape={self.mask_data.shape}, unique values={unique_values}")
            if not np.all(np.isin(unique_values, ORIGINAL_LABELS)):
                logger.warning(f"Mask contains values outside original labels for {self.mode}: {unique_values}")

        if self.use_unary:
            unique_values = np.unique(self.unary_data[0])
            logger.info(f"Unary data for {self.mode}: shape={self.unary_data.shape}, unique values={unique_values}")
            if not np.all(np.isin(unique_values, [0, 1])):
                logger.warning(f"Unary data contains non-binary values for {self.mode}: {unique_values}")
            unary_sum = np.sum(self.unary_data[0], axis=0).mean()
            logger.info(f"Unary data for {self.mode} channel sum mean: {unary_sum:.4f}")
        if self.return_weight:
            logger.info(f"Weight data for {self.mode}: shape={self.weight_data.shape}, range=[{self.weight_data[0].min():.4f}, {self.weight_data[0].max():.4f}]")

        self.num_samples = self.mri_data.shape[0]
        self.height, self.width = self.mri_data.shape[2], self.mri_data.shape[3]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {
            'mr': self.mri_data[idx][:],
            'mask': self.mask_data[idx][:] if self.mask_data is not None else None,
            'unary': self.unary_data[idx][:] if self.use_unary else None,
            'weight': self.weight_data[idx][:] if self.return_weight else None
        }

        if self.transform is not None and self.mode == 'train':
            sample = self._apply_transform(sample)

        sample['mr'] = torch.from_numpy(sample['mr']).float()
        if sample['mask'] is not None:
            sample['mask'] = torch.from_numpy(sample['mask']).long()
        if self.use_unary:
            sample['unary'] = torch.from_numpy(sample['unary'].copy()).float()
        if self.return_weight:
            sample['weight'] = torch.from_numpy(sample['weight']).float()

        if sample['mask'] is not None and sample['mask'].dim() == 3 and sample['mask'].size(0) == 1:
            sample['mask'] = sample['mask'].squeeze(0)

        if self.use_unary:
            if sample['mr'].dim() == 3 and sample['unary'].dim() == 3:
                inputs = torch.cat([sample['mr'], sample['unary']], dim=0)
                assert inputs.size(0) == 1 + len(ORIGINAL_LABELS), f"Expected {1 + len(ORIGINAL_LABELS)} channels, got {inputs.size(0)}"
            else:
                raise ValueError(f"Dimension mismatch: mr {sample['mr'].shape}, unary {sample['unary'].shape}")
        else:
            inputs = sample['mr']

        return inputs, sample['mask'], sample['unary'] if self.use_unary else None, idx

    def _apply_transform(self, sample):
        if self.transform is None or self.mode != 'train':
            return sample

        for key in sample.keys():
            if sample[key] is not None and isinstance(sample[key], np.ndarray):
                img = sample[key]
                if self.transform.get(key, None) is not None:
                    for t in self.transform[key]:
                        if t['name'] == 'RandomRotate':
                            angle = np.random.uniform(-t['angle_spectrum'], t['angle_spectrum'])
                            img = self._rotate(img, angle, interpolation=t.get('interpolation', 'nearest'))
                        elif t['name'] == 'ElasticDeformation':
                            img = self._elastic_deform(img, spline_order=t.get('spline_order', 3))
                        elif t['name'] == 'RandomContrast':
                            factor = np.random.uniform(0.8, 1.2)
                            img = np.clip(img * factor, 0, 1) if key == 'mr' else img
                        elif t['name'] == 'RandomFlip':
                            if np.random.random() < t.get('p', 0.5):
                                if img.ndim == 3:
                                    img = np.flip(img, axis=2).copy()
                                else:
                                    img = np.flip(img, axis=1).copy()
                        elif t['name'] == 'RandomBrightness':
                            factor = np.random.uniform(t.get('brightness_range', (0.8, 1.2))[0], t.get('brightness_range', (0.8, 1.2))[1])
                            img = np.clip(img * factor, 0, 1) if key == 'mr' else img
                sample[key] = img
        return sample

    def _rotate(self, img, angle, interpolation='nearest'):
        from scipy.ndimage import rotate
        if img.ndim == 3:
            return np.stack([rotate(img[c], angle, mode='constant', order=3 if interpolation == 'cubic' else 0, reshape=False) for c in range(img.shape[0])], axis=0)
        else:
            return rotate(img, angle, mode='constant', order=0 if interpolation == 'nearest' else 3, reshape=False)

    def _elastic_deform(self, img, spline_order=3):
        shape = img.shape[-2:]
        dx = gaussian_filter(np.random.randn(*shape), sigma=4, mode="reflect") * 2
        dy = gaussian_filter(np.random.randn(*shape), sigma=4, mode="reflect") * 2
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        if img.ndim == 3:
            deformed = np.stack([map_coordinates(img[c], indices, order=spline_order, mode='reflect').reshape(shape) for c in range(img.shape[0])], axis=0)
        else:
            deformed = map_coordinates(img, indices, order=spline_order, mode='reflect').reshape(shape)
        return deformed

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
            logger.info(f"HDF5 file closed for {self.mode} dataset")

def load_data(h5_file_path, batch_size=1, transformer_config=None, return_weight=False, num_classes=17, use_unary=False, skip_test_mask=False):
    def collate_fn(batch):
        imgs, masks, unaries, original_h5_indices = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        masks = None if all(m is None for m in masks) else torch.stack([m for m in masks if m is not None], dim=0)
        unaries = torch.stack(unaries, dim=0) if unaries[0] is not None else None
        original_h5_indices = torch.tensor(list(original_h5_indices), dtype=torch.long)
        return imgs, masks, unaries, original_h5_indices

    train_transform = transformer_config['train'] if transformer_config else None
    val_transform = transformer_config['val'] if transformer_config else None
    test_transform = transformer_config['test'] if transformer_config else None
    NUM_CLASSES = len(ORIGINAL_LABELS)
    train_dataset = SpineParseNetDataset(h5_file_path, mode='train', transform=train_transform, use_unary=use_unary,
                                         return_weight=return_weight, num_classes=NUM_CLASSES)

    val_dataset = SpineParseNetDataset(h5_file_path, mode='val', transform=val_transform, use_unary=use_unary,
                                      return_weight=return_weight, num_classes=num_classes)
    test_dataset = SpineParseNetDataset(h5_file_path, mode='test', transform=test_transform, use_unary=use_unary,
                                       return_weight=return_weight, num_classes=num_classes,
                                       skip_mask=skip_test_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_dataset.h5_file