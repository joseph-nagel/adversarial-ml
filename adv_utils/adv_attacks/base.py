'''Attack base class.'''

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn


class AdversarialAttack(nn.Module, ABC):
    '''Adversarial attack base class.'''

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | Callable[[torch.Tensor], torch.Tensor]
    ) -> None:

        super().__init__()

        self.model = model
        self.criterion = criterion

    @abstractmethod
    def forward(self):
        raise NotImplementedError
