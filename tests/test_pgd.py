import numpy as onp
import jax.numpy as np

from attax import pgd


def test_pgd():
    def f(x):
        return np.mean(x, axis=(1, 2))

    x = onp.random.rand(8, 32, 32, 3).astype(np.float32)
    y = onp.asarray(f(x).argmax(axis=-1))

    advs = pgd(f, x, y, epsilon=0.3)
    advs = onp.asarray(advs)

    perturbation = advs - x
    y_advs = onp.asarray(f(advs).argmax(axis=-1))

    assert x.shape == advs.shape
    assert onp.abs(perturbation).max() <= 0.3 + 1e7
    assert (y_advs == y).mean() < 1
