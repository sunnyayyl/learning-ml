from functools import partial

from jax import jit, random
from jaxtyping import jaxtyped
from typeguard import typechecked
import jax.numpy as jnp
from jaxtyping import ArrayLike, PRNGKeyArray, Float, Array
from typing import Iterable, Callable


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames=("shape", "y_function"))
def generate_data(
    key: ArrayLike,
    shape: Iterable[int],
    minval: ArrayLike,
    maxval: ArrayLike,
    y_function: Callable[[ArrayLike], ArrayLike],
    min_noise: float = 0.0,
    max_noise: float = 0.0,
) -> tuple[PRNGKeyArray, Float[Array, "input_size"], Float[Array, "input_size"]]:
    key, subkey, subkey1, subkey2, subkey3, subkey4 = random.split(key, 6)
    x = jnp.sort(random.uniform(key=subkey, shape=shape, minval=minval, maxval=maxval))

    y = y_function(x) + random.uniform(
        subkey4, shape, minval=min_noise, maxval=max_noise
    )
    x += +random.uniform(subkey1, shape, minval=min_noise, maxval=max_noise)

    return key, x, y
