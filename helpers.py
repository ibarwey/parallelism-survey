import flax
import optax
import jax
import jax.numpy as jnp
from flax.training import train_state
from typing import Callable
from Config import Config
from flax.training.common_utils import onehot


def loss_fn(logits, targets):
    loss = optax.softmax_cross_entropy(logits, onehot(targets, num_classes=Config.num_labels))
    return jnp.mean(loss)

def eval_fn(logits):
    return logits.argmax(-1)

class TrainState(train_state.TrainState):
    eval_function: Callable = flax.struct.field(pytree_node=False)
    loss_function: Callable = flax.struct.field(pytree_node=False)
