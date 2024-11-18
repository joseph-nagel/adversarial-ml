'''Fast-gradient sign method.'''

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from .base import AdversarialAttack


class FGSMAttack(AdversarialAttack):
    '''Fast gradient-sign attack.'''

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
    label: torch.Tensor | int | Sequence[int],
    eps: float,
    targeted: bool = False
) -> torch.Tensor:
    '''Perform a fast gradient-sign attack.'''

    # ensure tensor inputs
    perturbed = torch.as_tensor(image).detach().clone()

    label = torch.as_tensor(label).detach().clone()
    label = torch.atleast_1d(label)

    # disable param gradients
    param_requires_grad = []

    for p in model.parameters():
        param_requires_grad.append(p.requires_grad)
        p.requires_grad = False

    # enable input gradients
    perturbed.requires_grad = True

    # compute gradients
    pred = model(perturbed)
    loss = criterion(pred, label)

    loss.backward()
    grad = perturbed.grad.detach().clone()

    # disable input gradients
    perturbed.requires_grad = False

    # perform untargeted attack
    if not targeted:
        perturbed += eps * grad.sign()

    # perform targeted attack
    else:
        perturbed -= eps * grad.sign()

    # restore param gradients
    for p, p_requires_grad in zip(model.parameters(), param_requires_grad):
        p.requires_grad = p_requires_grad

    return perturbed

