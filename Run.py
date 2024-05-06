import jax
import jax.numpy as jnp
import optax
import flax
import time
from itertools import chain
from tqdm.notebook import tqdm

#from internal
from Config import Config
from DataProcessing import DataProcessing
from metrics import ACCURACY
from helpers import TrainState, loss_fn, eval_fn
from DataLoaders import sentimentTrainDataLoader, sentimentEvalDataLoader
from transformers import FlaxAutoModelForSequenceClassification

# FUNCTION DEFINITIONS--------------------------------------------------------------------------------------------------------------------------------------------
def train_step(state, batch, dropout_rng):
    targets = batch.pop("labels")
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_function(params):
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        loss = state.loss_function(logits, targets)
        return loss

    grad_fn = jax.value_and_grad(loss_function)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = jax.lax.pmean({'loss': loss, 'learning_rate': learning_rate_function(state.step)}, axis_name='batch')

    return new_state, metrics, new_dropout_rng

def eval_step(state, batch):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0]
    return state.eval_function(logits)


# BEGIN MAIN --------------------------------------------------------------------------------------------------------------------------------------------
# Initialize data processing
data_processor = DataProcessing('./data/train_file.csv', './data/test_file.csv')
train, valid, infer = data_processor.get_data_splits()

metric = ACCURACY()

num_train_steps = len(train) // Config.total_batch_size * Config.nb_epochs
learning_rate_function = optax.cosine_onecycle_schedule(transition_steps=num_train_steps, peak_value=Config.lr, pct_start=0.1)
print("The number of train steps (all the epochs) is", num_train_steps)

model = FlaxAutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis", from_pt=True)

optimizer = optax.adamw(learning_rate=Config.lr, b1=0.9, b2=0.999, eps=1e-6, weight_decay=1e-2)

#init trainstate
state = TrainState.create(
    apply_fn = model.__call__,
    params = model.params,
    tx = optimizer,
    eval_function=eval_fn,
    loss_function=loss_fn,
)

# NOTE: jax pmap for mapping function for every device
parallel_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))
parallel_eval_step = jax.pmap(eval_step, axis_name="batch")
parallel_inference = jax.pmap(eval_step, axis_name="batch")

# NOTE: replicates state (aka model) across all devices
state = flax.jax_utils.replicate(state)

# Setup training state
rng = jax.random.PRNGKey(Config.seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())

# TRAIN --------------------------------------------------------------------------------------------------------------------------------------------
def train():
    for i, epoch in enumerate(tqdm(range(1, Config.nb_epochs + 1), desc=f"Epoch...", position=0, leave=True)):
        rng, input_rng = jax.random.split(rng)

        # train
        with tqdm(total=len(train) // Config.total_batch_size, desc="Training...", leave=False) as progress_bar_train:
            for batch in sentimentTrainDataLoader(input_rng, train, Config.total_batch_size):
                state, train_metrics, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
                progress_bar_train.update(1)

        # evaluate
        with tqdm(total=len(valid) // Config.total_batch_size, desc="Evaluating...", leave=False) as progress_bar_eval:
            for batch in sentimentEvalDataLoader(valid, Config.total_batch_size):
                labels = batch.pop("labels")
                predictions = parallel_eval_step(state, batch)
                metric.add_batch(predictions=chain(*predictions), references=chain(*labels))
                progress_bar_eval.update(1)

        eval_metric = metric.compute()

        loss = round(flax.jax_utils.unreplicate(train_metrics)['loss'].item(), 3)
        eval_score = round(list(eval_metric.values())[0], 3)
        metric_name = list(eval_metric.keys())[0]

        print(f"{i+1}/{Config.nb_epochs} | Train loss: {loss} | Eval {metric_name}: {eval_score}")
    return state

state = train()

# INFERENCE --------------------------------------------------------------------------------------------------------------------------------------------
def inference(): # Perform inference
    start_time = time.time()
    with tqdm(total=len(infer) // Config.total_batch_size, desc="Inference...", leave=False) as progress_bar_inference:
        predictions = []
        for batch in sentimentEvalDataLoader(infer, Config.total_batch_size):
            labels = batch.pop("labels")
            preds = parallel_inference(state, batch)  # Use parallel inference function
            predictions.extend(preds)
            metric.add_batch(predictions=chain(*preds), references=chain(*labels))
            progress_bar_inference.update(1)
    end_time = time.time()

    # Combine predictions from all devices
    all_predictions = jnp.concatenate(predictions, axis=0)

    # Compute evaluation metric
    eval_metric = metric.compute()
    eval_score = round(list(eval_metric.values())[0], 3)
    metric_name = list(eval_metric.keys())[0]

    print(eval_score)
    return state

state = inference()