import jax.numpy as np
from jax.scipy.special import logsumexp


def crossentropy(logits, labels):
    logprobs = logits - logsumexp(logits, axis=1, keepdims=True)
    nll = np.take_along_axis(logprobs, np.expand_dims(labels, axis=1), axis=1)
    ce = -np.mean(nll)
    return ce
