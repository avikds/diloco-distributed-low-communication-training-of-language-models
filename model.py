"""
DiLoCo: Distributed Low-Communication Training of Language Models

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_model_params
import numpy as np

def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    """
    Initialize parameters for a 2-layer MLP.

    Shapes:
        W1: (input_dim, hidden_dim)
        b1: (hidden_dim,)
        W2: (hidden_dim, output_dim)
        b2: (output_dim,)

    All parameters are np.float64 arrays.
    Weights are initialized from a seeded NumPy RNG, while
    biases are initialized to zero.
    """
    rng = np.random.default_rng(seed)

    params = {
        "W1": rng.standard_normal((input_dim, hidden_dim)).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": rng.standard_normal((hidden_dim, output_dim)).astype(np.float64),
        "b2": np.zeros(output_dim, dtype=np.float64),
    }

    return params

# Step 2 - relu
def relu(x):
    """Apply element-wise Rectified Linear Unit (ReLU)."""
    return np.maximum(x, 0)

# Step 3 - model_forward
def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    z1 = x @ params["W1"] + params["b1"]
    h1 = relu(z1)
    logits = h1 @ params["W2"] + params["b2"]

    cache = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "logits": logits,
    }

    return logits, cache

# Step 4 - softmax
def softmax(logits):
    """Compute numerically stable row-wise softmax."""
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)

    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Step 5 - cross_entropy_loss
def cross_entropy_loss(logits, labels):
    """Compute mean numerically-stable cross-entropy loss."""
    # Numerically stable log-sum-exp:
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = (
        max_logits
        + np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))
    )

    # Log probability of the true class
    true_class_logits = logits[np.arange(logits.shape[0]), labels]
    true_class_log_probs = true_class_logits - log_sum_exp[:, 0]

    # Mean negative log-likelihood
    return -np.mean(true_class_log_probs)

# Step 6 - model_backward
def model_backward(params, cache, labels):
    """Compute gradients of the mean softmax cross-entropy loss."""
    x = cache["x"]
    z1 = cache["z1"]
    h1 = cache["h1"]
    logits = cache["logits"]

    batch_size = x.shape[0]

    # Softmax probabilities
    probs = softmax(logits)

    # Gradient of cross-entropy w.r.t. logits
    d_logits = probs.copy()
    d_logits[np.arange(batch_size), labels] -= 1.0
    d_logits /= batch_size

    # Output layer: logits = h1 @ W2 + b2
    dW2 = h1.T @ d_logits
    db2 = np.sum(d_logits, axis=0)

    # Backprop through output layer
    dh1 = d_logits @ params["W2"].T

    # Backprop through ReLU
    dz1 = dh1 * (z1 > 0)

    # Hidden layer: z1 = x @ W1 + b1
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }

# Step 7 - init_adamw_state
def init_adamw_state(params):
    """Initialize AdamW first/second moment state and step counter."""
    m = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    v = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    return {
        "m": m,
        "v": v,
        "t": 0,
    }

# Step 8 - update_adam_moments
def update_adam_moments(state, grads, beta1, beta2):
    """Update AdamW first and second raw moment estimates."""
    state["t"] += 1

    for key in grads:
        grad = grads[key]

        state["m"][key] = (
            beta1 * state["m"][key]
            + (1.0 - beta1) * grad
        )

        state["v"][key] = (
            beta2 * state["v"][key]
            + (1.0 - beta2) * (grad ** 2)
        )

    return state

# Step 9 - bias_correct_moments
def bias_correct_moments(state, beta1, beta2):
    """Return bias-corrected Adam first and second moments."""
    t = state["t"]

    m_correction = 1.0 - beta1 ** t
    v_correction = 1.0 - beta2 ** t

    m_hat = {
        key: value / m_correction
        for key, value in state["m"].items()
    }

    v_hat = {
        key: value / v_correction
        for key, value in state["v"].items()
    }

    return m_hat, v_hat

# Step 10 - adam_param_step
def adam_param_step(params, m_hat, v_hat, lr, eps):
    """Apply the core Adam adaptive parameter update without weight decay."""
    new_params = {}

    for key in params:
        new_params[key] = (
            params[key]
            - lr * m_hat[key] / (np.sqrt(v_hat[key]) + eps)
        )

    return new_params

# Step 11 - decoupled_weight_decay
def decoupled_weight_decay(params, lr, weight_decay):
    """Apply AdamW's decoupled weight decay without modifying params."""
    decay_factor = 1.0 - lr * weight_decay

    return {
        key: value * decay_factor
        for key, value in params.items()
    }

# Step 12 - clone_params
def clone_params(params):
    """Return a deep copy of the parameter arrays."""
    return {
        key: value.copy()
        for key, value in params.items()
    }

# Step 13 - scale_params
def scale_params(params, scalar):
    """Return a new parameter dictionary with every array scaled by scalar."""
    return {
        key: value * scalar
        for key, value in params.items()
    }

# Step 14 - subtract_params
def subtract_params(params_a, params_b):
    """Return the element-wise difference params_a - params_b."""
    return {
        key: params_a[key] - params_b[key]
        for key in params_a
    }

# Step 15 - average_params
def average_params(params_list):
    """Return the element-wise mean of a non-empty list of parameter dicts."""
    if not params_list:
        raise ValueError("params_list must be non-empty")

    keys = params_list[0].keys()
    n = len(params_list)

    return {
        key: sum(params[key] for params in params_list) / n
        for key in keys
    }

# Step 16 - iid_shard_dataset
def iid_shard_dataset(x, y, num_workers, seed=0):
    """Partition (x, y) into reproducible, disjoint IID shards."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of examples")

    rng = np.random.default_rng(seed)

    # Shuffle indices so each worker receives a random IID subset.
    indices = rng.permutation(len(x))

    # Remainder examples go to the earliest shards.
    base_size = len(x) // num_workers
    remainder = len(x) % num_workers

    shards = []
    start = 0

    for worker_idx in range(num_workers):
        shard_size = base_size + (1 if worker_idx < remainder else 0)
        shard_indices = indices[start:start + shard_size]

        shards.append((
            x[shard_indices],
            y[shard_indices],
        ))

        start += shard_size

    return shards

# Step 17 - noniid_shard_dataset
def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    """Partition (x, y) into reproducible non-IID class-based worker shards."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of examples")

    rng = np.random.default_rng(seed)

    shards = []

    for worker_idx in range(num_workers):
        # Worker owns classes c where c % num_workers == worker_idx.
        assigned_classes = np.arange(worker_idx, num_classes, num_workers)

        # Select all examples belonging to this worker's classes.
        mask = np.isin(y, assigned_classes)
        indices = np.flatnonzero(mask)

        # Shuffle examples within the worker's shard.
        rng.shuffle(indices)

        shards.append((
            x[indices],
            y[indices],
        ))

    return shards

# Step 18 - sample_worker_batch
def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    """Sample a random mini-batch from a worker's local shard."""
    n = len(x_shard)

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")

    if n == 0 and batch_size > 0:
        raise ValueError("cannot sample from an empty shard")

    replace = batch_size > n

    indices = rng.choice(
        n,
        size=batch_size,
        replace=replace,
    )

    return x_shard[indices], y_shard[indices]

# Step 19 - local_train_step
def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    """Perform one complete AdamW training step on a mini-batch."""
    logits, cache = model_forward(params, x_batch)
    loss = cross_entropy_loss(logits, y_batch)
    grads = model_backward(params, cache, y_batch)
    new_adam_state = update_adam_moments(adam_state, grads, beta1, beta2)
    m_hat, v_hat = bias_correct_moments(new_adam_state, beta1, beta2)
    new_params = adam_param_step(params, m_hat, v_hat, lr, eps)
    new_params = decoupled_weight_decay(new_params, lr, weight_decay)
    return new_params, new_adam_state, loss

# Step 20 - inner_train_worker
def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    """Run the local AdamW inner loop for one DiLoCo worker."""
    worker_params = clone_params(params)
    adam_state = init_adamw_state(worker_params)
    rng = np.random.default_rng(seed)

    losses = []

    for _ in range(num_inner_steps):
        x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)

        worker_params, adam_state, loss = local_train_step(
            worker_params, adam_state, x_batch, y_batch,
            lr, beta1, beta2, eps, weight_decay
        )

        losses.append(float(loss))

    mean_loss = float(np.mean(losses))

    return worker_params, mean_loss

# Step 21 - init_outer_optimizer
def init_outer_optimizer(params):
    """Initialize the server-side Nesterov momentum state."""
    momentum = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    return {
        "momentum": momentum
    }

# Step 22 - update_outer_momentum
def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    """Update Nesterov momentum buffer: m <- momentum_coef * m + outer_grad."""
    for key in outer_state["momentum"]:
        outer_state["momentum"][key] = (
            momentum_coef * outer_state["momentum"][key]
            + outer_grad[key]
        )

    return outer_state

# Step 23 - nesterov_param_update (not yet solved)
# TODO: implement

# Step 24 - compute_outer_gradient (not yet solved)
# TODO: implement

# Step 25 - run_diloco_round (not yet solved)
# TODO: implement

# Step 26 - train_diloco (not yet solved)
# TODO: implement

# Step 27 - train_synchronous_baseline (not yet solved)
# TODO: implement

# Step 28 - evaluate_loss (not yet solved)
# TODO: implement

# Step 29 - classification_accuracy (not yet solved)
# TODO: implement

# Step 30 - communication_savings (not yet solved)
# TODO: implement

