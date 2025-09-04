'''Adv. training base class.'''

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from ..adv_attacks import AdversarialAttack


class AdversarialTraining(LightningModule):
    '''
    Adversarial training.

    Parameters
    ----------
    model : PyTorch module
        Trainable model.
    criterion : PyTorch module or callable
        Standard loss function.
    attack : AdversarialAttack
        Adversarial attack.
    alpha : float
        Weighting parameter.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        attack: AdversarialAttack,
        alpha: float = 0.5,
        lr: float = 1e-04
    ) -> None:

        super().__init__()

        # set model
        self.model = model

        # set loss function
        self.criterion = criterion

        # set adversarial attack
        self.attack = attack

        # set weight parameter
        self.alpha = min(1., max(0., alpha))

        # set initial learning rate
        self.lr = abs(lr)

        # store hyperparams
        self.save_hyperparameters(
            ignore=['model', 'criterion', 'attack'],
            logger=True
        )

    @property
    def std_weight(self):
        return self.alpha

    @property
    def adv_weight(self):
        return 1 - self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Run model.'''
        return self.model(x)

    def standard_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''Compute standard loss.'''
        y_pred = self.model(x)
        return self.criterion(y_pred, y)

    def adversarial_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''Compute adversarial loss.'''

        # perform attack (with gradients enabled)
        with torch.enable_grad():
            x_adv = self.attack(x, y)

        return self.standard_loss(x_adv, y)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''Compute loss.'''

        # compute standard loss
        std_loss = self.standard_loss(x, y) if self.std_weight > 0. else 0.0

        # compute adversarial loss
        adv_loss = self.adversarial_loss(x, y) if self.adv_weight > 0. else 0.0

        return self.std_weight * std_loss + self.adv_weight * adv_loss

    @staticmethod
    def _get_batch(
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''Get batch features and labels.'''

        if isinstance(batch, Sequence):
            x_batch = batch[0]
            y_batch = batch[1]

        elif isinstance(batch, dict):
            x_batch = batch['images']
            y_batch = batch['labels']

        else:
            raise TypeError(f'Invalid batch type: {type(batch)}')

        return x_batch, y_batch

    def training_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)

        self.log('train_loss', loss.item())  # Lightning logs batch-wise scalars during training per default

        return loss

    def validation_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)

        self.log('val_loss', loss.item())  # Lightning automatically averages scalars over batches for validation

        return loss

    def test_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss = self.loss(x_batch, y_batch)

        self.log('test_loss', loss.item())  # Lightning automatically averages scalars over batches for testing

        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
