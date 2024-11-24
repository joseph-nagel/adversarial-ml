'''Adv. classifier training.'''

from collections.abc import Sequence

import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy

from ..hugging import HFClassifier
from ..adv_attacks import FGSMAttack
from .base import AdversarialTraining


class AdversarialHFClassifier(AdversarialTraining):
    '''Adv. training of a pretrained classifier.'''

    def __init__(
        self,
        model_name: str = 'google/vit-base-patch16-224',
        data_dir: str | None = None,
        num_labels: int = 10,
        freeze_features: bool = False,
        eps: float = 0.003,
        targeted: bool = False,
        alpha: float = 0.5,
        lr: float = 1e-04
    ) -> None:

        # load pretrained model
        model = HFClassifier.from_pretrained(
            model_name,
            cache_dir=data_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=False if num_labels is None else True
        )

        # freeze/unfreeze parameters
        if freeze_features:
            for p in model.parameters():
                p.requires_grad = False

            for p in model.model.classifier.parameters():
                p.requires_grad = True

        else:
            for p in model.parameters():
                p.requires_grad = True

        # create criterion
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # create attack
        attack = FGSMAttack(
            model=model,
            criterion=criterion,
            eps=eps,
            targeted=targeted
        )

        # initialize parent class
        super().__init__(
            model=model,
            criterion=criterion,
            attack=attack,
            alpha=alpha,
            lr=lr
        )

        # create accuracy metrics
        if self.std_weight > 0.:
            self.std_train_acc = Accuracy(task='multiclass', num_classes=num_labels)
            self.std_val_acc = Accuracy(task='multiclass', num_classes=num_labels)
            self.std_test_acc = Accuracy(task='multiclass', num_classes=num_labels)

        if self.adv_weight > 0.:
            self.adv_train_acc = Accuracy(task='multiclass', num_classes=num_labels)
            self.adv_val_acc = Accuracy(task='multiclass', num_classes=num_labels)
            self.adv_test_acc = Accuracy(task='multiclass', num_classes=num_labels)

    def standard_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_pred: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Compute standard loss.'''

        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        if return_pred:
            return loss, y_pred
        else:
            return loss

    def adversarial_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_pred: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Compute adversarial loss.'''

        # perform attack (with gradients enabled)
        with torch.enable_grad():
            x_adv = self.attack(x, y)

        return self.standard_loss(x_adv, y, return_pred)

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_pred: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Compute loss.'''

        # compute standard loss
        std_out = self.standard_loss(x, y, return_pred) if self.std_weight > 0. else 0.0

        if isinstance(std_out, tuple):
            std_loss, std_pred = std_out
        else:
            std_loss = std_out
            std_pred = None

        # compute adversarial loss
        adv_out = self.adversarial_loss(x, y, return_pred) if self.adv_weight > 0. else 0.0

        if isinstance(adv_out, tuple):
            adv_loss, adv_pred = adv_out
        else:
            adv_loss = adv_out
            adv_pred = None

        # compute total loss
        loss = self.std_weight * std_loss + self.adv_weight * adv_loss

        if return_pred:
            return loss, std_pred, adv_pred
        else:
            return loss

    def training_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss, std_pred, adv_pred = self.loss(x_batch, y_batch, return_pred=True)

        if hasattr(self, 'std_train_acc') and std_pred is not None:
            _ = self.std_train_acc(std_pred, y_batch)
            self.log('std_train_acc', self.std_train_acc)

        if hasattr(self, 'adv_train_acc') and adv_pred is not None:
            _ = self.adv_train_acc(adv_pred, y_batch)
            self.log('adv_train_acc', self.adv_train_acc)

        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default

        return loss

    def validation_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss, std_pred, adv_pred = self.loss(x_batch, y_batch, return_pred=True)

        if hasattr(self, 'std_val_acc') and std_pred is not None:
            _ = self.std_val_acc(std_pred, y_batch)
            self.log('std_val_acc', self.std_val_acc)

        if hasattr(self, 'adv_val_acc') and adv_pred is not None:
            _ = self.adv_val_acc(adv_pred, y_batch)
            self.log('adv_val_acc', self.adv_val_acc)

        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation

        return loss

    def test_step(
        self,
        batch: Sequence[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        x_batch, y_batch = self._get_batch(batch)
        loss, std_pred, adv_pred = self.loss(x_batch, y_batch, return_pred=True)

        if hasattr(self, 'std_test_acc') and std_pred is not None:
            _ = self.std_test_acc(std_pred, y_batch)
            self.log('std_test_acc', self.std_test_acc)

        if hasattr(self, 'adv_test_acc') and adv_pred is not None:
            _ = self.adv_test_acc(adv_pred, y_batch)
            self.log('adv_test_acc', self.adv_test_acc)

        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing

        return loss

