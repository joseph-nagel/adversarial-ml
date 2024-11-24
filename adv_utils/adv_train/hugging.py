'''Adv. classifier training.'''

import torch.nn as nn

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
        for p in model.parameters():
            p.requires_grad = False

        for p in model.model.classifier.parameters():
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

