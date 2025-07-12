'''Flowers102 datamodule.'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning import LightningDataModule


# define type alias
FloatOrFloats = float | tuple[float, float, float]


class Flowers102DataModule(LightningDataModule):
    '''
    DataModule for the Flowers102 dataset.

    Parameters
    ----------.
    data_dir : str
        Directory for storing the data.
    mean : float, (float, float, float) or None
        Mean for data normalization.
    std : float, (float, float, float) or None
        Standard deviation for normalization.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(
        self,
        data_dir: str = '.',
        mean: FloatOrFloats | None = (0.5, 0.5, 0.5),
        std: FloatOrFloats | None = (0.5, 0.5, 0.5),
        batch_size: int = 32,
        num_workers: int = 0
    ) -> None:

        super().__init__()

        # set data location
        self.data_dir = data_dir

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # create transforms
        train_transforms = [
            transforms.RandomRotation(45),  # TODO: refine data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToTensor()
        ]

        test_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]

        if (mean is not None) and (std is not None):
            normalize = transforms.Normalize(mean=mean, std=std)

            train_transforms.append(normalize)
            test_transforms.append(normalize)

        self.train_transform = transforms.Compose(train_transforms)
        self.test_transform = transforms.Compose(test_transforms)

        # create inverse normalization
        if (mean is not None) and (std is not None):

            mean = torch.as_tensor(mean).view(-1, 1, 1)
            std = torch.as_tensor(std).view(-1, 1, 1)

            self.renormalize = transforms.Compose([
                transforms.Lambda(lambda x: x * std + mean),  # reverse normalization
                transforms.Lambda(lambda x: x.clamp(0, 1))  # clip to valid range
            ])

    def prepare_data(self) -> None:
        '''Download data.'''

        train_set = datasets.Flowers102(
            self.data_dir,
            split='train',
            download=True
        )

        val_set = datasets.Flowers102(
            self.data_dir,
            split='val',
            download=True
        )

        test_set = datasets.Flowers102(
            self.data_dir,
            split='test',
            download=True
        )

    def setup(self, stage: str) -> None:
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            self.train_set = datasets.Flowers102(
                self.data_dir,
                split='train',
                transform=self.train_transform
            )

            self.val_set = datasets.Flowers102(
                self.data_dir,
                split='val',
                transform=self.train_transform
            )

        # create test dataset
        elif stage == 'test':
            self.test_set = datasets.Flowers102(
                self.data_dir,
                split='test',
                transform=self.test_transform
            )

    def train_dataloader(self) -> DataLoader:
        '''Create train dataloader.'''

        if hasattr(self, 'train_set'):
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Train set has not been set')

    def val_dataloader(self) -> DataLoader:
        '''Create val. dataloader.'''

        if hasattr(self, 'val_set'):
            return DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Val. set has not been set')

    def test_dataloader(self) -> DataLoader:
        '''Create test dataloader.'''

        if hasattr(self, 'test_set'):
            return DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Test set has not been set')

