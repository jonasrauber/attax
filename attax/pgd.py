import jax.numpy as np
from jax import grad
from jax import random
from .utils import crossentropy


def pgd(
    f,
    x,
    y,
    *,
    epsilon=0.3,
    num_steps=40,
    step_size=0.01,
    random_start=True,
    seed=0,
    bounds=(0, 1),
):
    """L-infinity Projected Gradient Descent Attack"""

    key = random.PRNGKey(seed)

    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape[0] == y.shape[0]
    assert y.ndim == 1

    x0 = x.copy()

    def loss(x):
        logits = f(x)
        return crossentropy(logits, y)

    g = grad(loss)

    if random_start:
        x = x + random.uniform(key, x.shape, minval=-epsilon, maxval=epsilon)
        x = np.clip(x, *bounds)

    for _ in range(num_steps):
        gradient = g(x)
        x = x + step_size * np.sign(gradient)
        x = np.clip(x, x0 - epsilon, x0 + epsilon)
        x = np.clip(x, *bounds)

    return x
