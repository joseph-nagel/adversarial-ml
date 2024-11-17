'''Utilities.'''

from collections.abc import Callable, Sequence

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
        eps: float,
        targeted: bool = False
    ) -> torch.Tensor:

        return fgsm_attack(
            model=self.model,
            criterion=self.criterion,
            image=image,
            label=label,
            eps=eps,
            targeted=targeted
        )


def fgsm_attack(
    model: nn.Module,
    criterion: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    label: torch.Tensor | Sequence[int],
    eps: float,
    targeted: bool = False
) -> torch.Tensor:
    '''Perform a fast gradient-sign attack.'''

    # ensure tensor inputs
    image = torch.as_tensor(image).detach().clone()
    label = torch.as_tensor(label).detach().clone()

    # enable input gradients
    image.requires_grad = True

    # disable param gradients
    param_requires_grad = []

    for p in model.parameters():
        param_requires_grad.append(p.requires_grad)
        p.requires_grad = False

    # compute gradients
    pred = model(image)

    loss = criterion(pred, label)

    loss.backward()

    grad = image.grad.detach().clone()

    # perturb input
    perturbed = image.detach().clone()

    if not targeted:
        # perform untargeted attack
        perturbed += eps * grad.sign()

    else:
        # perform targeted attack
        perturbed -= eps * grad.sign()

    # restore param gradients
    for p, p_requires_grad in zip(model.parameters(), param_requires_grad):
        p.requires_grad = p_requires_grad

    return perturbed

