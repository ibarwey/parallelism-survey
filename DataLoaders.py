import jax
import jax.numpy as jnp
import flax
from flax.training.common_utils import shard


def sentimentTrainDataLoader(rng, dataset, batch_size):
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch

def sentimentEvalDataLoader(dataset, batch_size):
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        #sharding
        batch = shard(batch)

        yield batch