'''Utilities.'''

from collections.abc import Callable

import torch
import torch.nn as nn


class FGSMAttack(nn.Module):
    '''Fast gradient-sign attack.'''

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | Callable[[torch.Tensor], torch.Tensor]
    ) -> None:

        super().__init__()

        self.model = model
        self.criterion = criterion

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        eps: float
    ) -> torch.Tensor:

        return fgsm_attack(
            model = self.model,
            criterion = self.criterion,
            image = image,
            label = label,
            eps = eps
        )


def fgsm_attack(
    model: nn.Module,
    criterion: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    label: torch.Tensor,
    eps: float
) -> torch.Tensor:
    '''Perform a fast gradient-sign attack.'''

    # enable input gradients
    image_requires_grad = image.requires_grad
    image.requires_grad = True

    # disable param gradients
    param_requires_grad = []

    for p in model.parameters():
        param_requires_grad.append(p.requires_grad)
        p.requires_grad = False

    # reset gradients
    image.grad = None
    model.zero_grad()

    # compute gradients
    pred = model(image)

    loss = criterion(pred, label)

    loss.backward()

    grad = image.grad.detach().clone()

    # perturb input
    perturbed = image.detach() + eps * grad.sign()

    # restore input gradients
    image.requires_grad = image_requires_grad

    # restore param gradients
    for p, p_requires_grad in zip(model.parameters(), param_requires_grad):
        p.requires_grad = p_requires_grad

    return perturbed

