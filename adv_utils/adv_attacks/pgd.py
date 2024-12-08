'''Projected gradient descent.'''

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from .base import AdversarialAttack


class PGDAttack(AdversarialAttack):
    '''Projected gradient descent attack.'''

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        num_steps: int,
        step_size: float,
        eps: float,
        p_norm: int | float = torch.inf,
        targeted: bool = False
    ) -> None:

        super().__init__(model, criterion)

        self.num_steps = abs(num_steps)
        self.step_size = abs(step_size)
        self.eps = abs(eps)
        self.p_norm = p_norm
        self.targeted = targeted

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:

        return pgd_attack(
            model=self.model,
            criterion=self.criterion,
            image=image,
            label=label,
            num_steps=self.num_steps,
            step_size=self.step_size,
            eps=self.eps,
            p_norm=self.p_norm,
            targeted=self.targeted
        )


def pgd_attack(
    model: nn.Module,
    criterion: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    label: torch.Tensor | int | Sequence[int],
    num_steps: int,
    step_size: float,
    eps: float,
    p_norm: int | float = torch.inf,
    targeted: bool = False
) -> torch.Tensor:
    '''Perform a projected gradient descent attack.'''

    # ensure tensor inputs
    perturbed = torch.as_tensor(image).detach().clone()

    label = torch.as_tensor(label).detach().clone()
    label = torch.atleast_1d(label)

    # disable param gradients
    param_requires_grad = []

    for p in model.parameters():
        param_requires_grad.append(p.requires_grad)
        p.requires_grad = False

    # perform iterations
    for _ in range(num_steps):

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
            perturbed += step_size * grad

        # perform targeted attack
        else:
            perturbed -= step_size * grad

        # rescale (if outside of l2-ball)
        if p_norm == 2:
            delta = perturbed - image # (b, 3, h, w)

            norm = torch.linalg.vector_norm(
                delta,
                dim=tuple(range(1, perturbed.ndim)),
                keepdim=True
            ) # (b, 1, 1, 1)

            delta = torch.where(norm > eps, delta / norm, delta) # (b, 3, h, w)

            perturbed = image + delta # (b, 3, h, w)

        # clip (each dimension)
        elif p_norm == torch.inf:
            perturbed = perturbed.clamp(image - eps, image + eps) # (b, 3, h, w)

        else:
            raise ValueError(f'Unsupported p-norm: {p_norm}')

    # restore param gradients
    for p, p_requires_grad in zip(model.parameters(), param_requires_grad):
        p.requires_grad = p_requires_grad

    return perturbed

