====================================
Attax: adversarial attacks using JAX
====================================

Attax is python library with implementations of some common adversarial attacks
using JAX. For now this is a prototype at best and I refer you my other project,
`Foolbox <https://github.com/bethgelab/foolbox>`_, that also supports JAX and
provides a well-tested suite of adversarial attacks.

Installation
------------

.. code-block:: bash

   pip install attax


Example
-------

.. code-block:: python

   import attax
   from functools import partial

   def predict(params, inputs):
       # see https://github.com/google/jax
       # ...

   params = ...  # model parameters
   x = ...  # input data
   y = ...  # labels

   f = partial(predict, params)

   x_adv = attax.pgd(f, x, y, epsilon=0.3)

   print((f(x).argmax(axis=-1) == y).mean())
   print((f(x_adv).argmax(axis=-1) == y).mean())
