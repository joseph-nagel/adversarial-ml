'''Random sampling.'''

from collections.abc import Sequence

import torch


def sample_interval(
    size: int | Sequence[int],
    interval: tuple[float, float] = (0., 1.),
    dtype: torch.dtype | None = None
):
    '''Sample uniformly over an interval.'''

    # get lower/upper bound
    if not isinstance(interval, Sequence):
        raise TypeError(f'Invalid interval type: {type(interval)}')

    elif len(interval) != 2:
        raise ValueError(f'Invalued number of elements: {len(interval)}')

    lower = min(interval)
    upper = max(interval)

    # sample uniformly in [0, 1]
    u = torch.rand(size, dtype=dtype)

    # transform to target interval
    samples = (upper - lower) * u + lower

    return samples


def sample_sphere(
    num_samples: int,
    num_dim: int = 3,
    radius: float = 1.,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    '''
    Sample uniformly on an (n-1)-sphere.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    num_dim : int
        Ambient space dimensionality.
    radius : float
        Target radius.
    dtype : torch.dtype or None
        Data type.

    '''

    # generate standard normal samples
    samples = torch.randn((num_samples, num_dim), dtype=dtype)

    # calculate lengths
    # lengths = torch.linalg.vector_norm(samples, ord=2, dim=1, keepdim=True)
    lengths = torch.sqrt(torch.sum(samples**2, dim=1, keepdim=True))

    # normalize to unit length
    samples /= lengths

    # scale to target radius
    samples *= radius

    return samples.squeeze()


def sample_ball(
    num_samples: int,
    num_dim: int = 3,
    radius: float = 1.,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    '''
    Sample uniformly within an n-ball.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    num_dim : int
        Ambient space dimensionality.
    radius : float
        Target radius.
    dtype : torch.dtype or None
        Data type.

    '''

    # sample uniformly on the unit sphere
    samples_on_sphere = sample_sphere(
        num_samples,
        num_dim=num_dim,
        radius=1.,
        dtype=dtype
    )

    # sample uniformly in [0, 1]
    u = torch.rand((num_samples, 1), dtype=dtype)

    # create samples within ball
    samples_in_ball = samples_on_sphere * torch.pow(u, 1 / num_dim)

    # scale to target radius
    samples_in_ball *= radius

    return samples_in_ball.squeeze()

