import jax

class Config:
    nb_epochs = 2
    lr = 2e-5
    per_device_bs = 32
    num_labels = 3
    seed = 42
    total_batch_size = per_device_bs * jax.local_device_count()
